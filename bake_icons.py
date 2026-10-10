#!/usr/bin/env python3
"""
Bakes Blender icon SVGs into the project's square icon viewBox (24x24 by default).

Every transform in a source file (group and element transforms, including rotations, mirrors, and
non-uniform scales) is folded into the path coordinates, so the output carries no transform
attributes at all.  Basic shapes (rect, circle, ellipse, line, polyline, polygon) become paths, since
the editor's PathParser reads path data only; arcs become cubic Béziers (see arcs_to_cubics).  Stroke
widths and userSpaceOnUse gradients follow the baked transform.  Editor metadata (sodipodi / inkscape) is dropped.

Placement: Blender authors its icons at 100 units per icon pixel on a 1600-unit cell, and the
source viewBoxes are that cell cropped or padded per icon (1600x1400, 1800x1600, ...).  The default
fit maps 1600 units onto the output size for every icon, so all icons keep Blender's relative scale,
and centers each source viewBox (not the art's bounds) in the output box, which keeps icon families
aligned (mouse_lmb's mouse sits where mouse_lmb_drag's does).  The few icons whose art is wider than
the box (keyboard keys, the wordmark) shrink about the box center until they fit; see --fit and
--overflow for the alternatives.

Usage:
	python3 bake_icons.py SOURCE_DIR DEST_DIR [--size 24] [--unit 1600] [--fit unit|box] [--overflow shrink|keep]
"""

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"
EDITOR_NAMESPACES = (
	"http://www.inkscape.org/namespaces/inkscape",
	"http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
	"http://www.serif.com/",
	"http://creativecommons.org/ns#",
	"http://purl.org/dc/elements/1.1/",
	"http://www.w3.org/1999/02/22-rdf-syntax-ns#",
)
IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

# Elements carried through with their children walked.
CONTAINER_TAGS = {"g", "a", "switch"}
# Elements converted to a path.
SHAPE_TAGS = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"}
# Elements copied verbatim (their coordinates do not live in the referencing element's user space,
# or they carry no geometry).
PASSIVE_TAGS = {"defs", "linearGradient", "radialGradient", "stop", "title", "desc", "style"}
# Elements dropped outright.
DROPPED_TAGS = {"metadata"}
# Geometry attributes a shape loses once it is a path.
SHAPE_GEOMETRY_ATTRIBUTES = {"x", "y", "width", "height", "rx", "ry", "cx", "cy", "r", "x1", "y1", "x2", "y2", "points"}
# Properties that inherit down the tree and matter to the bake.
INHERITED_PROPERTIES = ("stroke", "stroke-width", "stroke-dasharray", "stroke-dashoffset", "fill", "fill-opacity", "stroke-opacity")
# Attributes with no effect on the rendered icon.
NOISE_ATTRIBUTES = {"enable-background", "version", "{http://www.w3.org/XML/1998/namespace}space"}

NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
TRANSFORM_PATTERN = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)")
URL_PATTERN = re.compile(r"url\(\s*['\"]?#([^)'\"]+)['\"]?\s*\)")


class IconError(Exception):
	"""A source file this tool cannot bake faithfully."""


# ---------------------------------------------------------------------------------------------------
# Affine algebra.  A matrix is (a, b, c, d, e, f) with SVG semantics: x' = a x + c y + e,
# y' = b x + d y + f.


def multiply(outer, inner):
	"""
	Composes two affine matrices so that inner applies first, then outer.

	@param tuple outer The matrix applied second.
	@param tuple inner The matrix applied first.
	@return tuple The composed matrix.
	"""
	a1, b1, c1, d1, e1, f1 = outer
	a2, b2, c2, d2, e2, f2 = inner
	return (
		a1 * a2 + c1 * b2,
		b1 * a2 + d1 * b2,
		a1 * c2 + c1 * d2,
		b1 * c2 + d1 * d2,
		a1 * e2 + c1 * f2 + e1,
		b1 * e2 + d1 * f2 + f1,
	)


def apply_point(matrix, x, y):
	"""
	Maps one point through an affine matrix.

	@param tuple matrix The transform.
	@param float x      Source x.
	@param float y      Source y.
	@return tuple The mapped (x, y).
	"""
	a, b, c, d, e, f = matrix
	return (a * x + c * y + e, b * x + d * y + f)


def determinant(matrix):
	"""
	Returns the determinant of a matrix's linear part (negative means the transform mirrors).

	@param tuple matrix The transform.
	@return float The determinant.
	"""
	a, b, c, d, _, _ = matrix
	return a * d - b * c


def axis_scales(matrix):
	"""
	Returns the singular values of a matrix's linear part: how far it stretches along its major and
	minor axes.  Equal values mean the transform scales uniformly (plus any rotation or mirror).

	@param tuple matrix The transform.
	@return tuple (major, minor) stretch factors.
	"""
	a, b, c, d, _, _ = matrix
	sum_of_squares = a * a + b * b + c * c + d * d
	det = abs(a * d - b * c)
	discriminant = math.sqrt(max(sum_of_squares * sum_of_squares / 4.0 - det * det, 0.0))
	major = math.sqrt(max(sum_of_squares / 2.0 + discriminant, 0.0))
	minor = math.sqrt(max(sum_of_squares / 2.0 - discriminant, 0.0))
	return (major, minor)


def parse_transform(text):
	"""
	Parses an SVG transform list into one matrix.

	@param str text The transform attribute value.
	@return tuple The composed matrix (leftmost entry applied last, per SVG).
	"""
	result = IDENTITY
	consumed = TRANSFORM_PATTERN.sub("", text).replace(",", " ").strip()
	if consumed:
		raise IconError(f"unparsed transform text: {consumed!r}")
	for name, arguments in TRANSFORM_PATTERN.findall(text):
		values = [float(value) for value in NUMBER_PATTERN.findall(arguments)]
		if name == "matrix":
			if len(values) != 6:
				raise IconError(f"matrix() needs 6 values: {text!r}")
			step = tuple(values)
		elif name == "translate":
			step = (1.0, 0.0, 0.0, 1.0, values[0], values[1] if len(values) > 1 else 0.0)
		elif name == "scale":
			scale_x = values[0]
			scale_y = values[1] if len(values) > 1 else scale_x
			step = (scale_x, 0.0, 0.0, scale_y, 0.0, 0.0)
		elif name == "rotate":
			radians = math.radians(values[0])
			cosine = math.cos(radians)
			sine = math.sin(radians)
			step = (cosine, sine, -sine, cosine, 0.0, 0.0)
			if len(values) == 3:
				center_x, center_y = values[1], values[2]
				step = multiply((1.0, 0.0, 0.0, 1.0, center_x, center_y), multiply(step, (1.0, 0.0, 0.0, 1.0, -center_x, -center_y)))
		elif name == "skewX":
			step = (1.0, 0.0, math.tan(math.radians(values[0])), 1.0, 0.0, 0.0)
		else:
			step = (1.0, math.tan(math.radians(values[0])), 0.0, 1.0, 0.0, 0.0)
		result = multiply(result, step)
	return result


# ---------------------------------------------------------------------------------------------------
# Path data.  Segments are absolute: ("M", x, y), ("L", x, y), ("C", x1, y1, x2, y2, x, y),
# ("Q", x1, y1, x, y), ("A", rx, ry, rotation, large, sweep, x, y), ("Z",).  Arcs exist only between
# parsing and arcs_to_cubics.


class PathScanner:
	"""Character-level reader for path data (handles packed numbers like "1.5.5" and arc flags)."""

	def __init__(self, text):
		"""
		Starts a scan over one path's data.

		@param str text The d attribute.
		"""
		self.text = text
		self.position = 0

	def skip_separators(self):
		"""
		Advances past whitespace and commas.
		"""
		while self.position < len(self.text) and self.text[self.position] in " \t\r\n,":
			self.position += 1

	def at_end(self):
		"""
		Reports whether only separators remain.

		@return bool True when the data is exhausted.
		"""
		self.skip_separators()
		return self.position >= len(self.text)

	def peek_command(self):
		"""
		Returns the next command letter without consuming it, or None if a number comes next.

		@return str|None The command letter.
		"""
		self.skip_separators()
		if self.position < len(self.text) and self.text[self.position].isalpha():
			return self.text[self.position]
		return None

	def read_command(self):
		"""
		Consumes and returns one command letter.

		@return str The command letter.
		"""
		command = self.peek_command()
		if command is None:
			raise IconError(f"expected a path command at {self.position}: {self.text[self.position:self.position + 20]!r}")
		self.position += 1
		return command

	def read_number(self):
		"""
		Consumes and returns one number.

		@return float The number.
		"""
		self.skip_separators()
		match = NUMBER_PATTERN.match(self.text, self.position)
		if match is None:
			raise IconError(f"expected a number at {self.position}: {self.text[self.position:self.position + 20]!r}")
		self.position = match.end()
		return float(match.group(0))

	def read_flag(self):
		"""
		Consumes one arc flag, which is a single 0 or 1 character that may abut the next number.

		@return int The flag.
		"""
		self.skip_separators()
		if self.position >= len(self.text) or self.text[self.position] not in "01":
			raise IconError(f"expected an arc flag at {self.position}")
		flag = int(self.text[self.position])
		self.position += 1
		return flag


def parse_path(text):
	"""
	Parses path data into absolute segments, expanding H/V to L, S to C, and T to Q.

	@param str text The d attribute.
	@return list The absolute segments.
	"""
	scanner = PathScanner(text)
	segments = []
	current_x = current_y = 0.0
	start_x = start_y = 0.0
	last_cubic_control = None
	last_quadratic_control = None
	command = None
	while not scanner.at_end():
		next_command = scanner.peek_command()
		if next_command is not None:
			command = scanner.read_command()
		elif command is None:
			raise IconError("path data does not start with a command")
		elif command in "Mm":
			# Extra coordinate pairs after a moveto are implicit linetos.
			command = "L" if command == "M" else "l"
		relative = command.islower()
		upper = command.upper()
		base_x = current_x if relative else 0.0
		base_y = current_y if relative else 0.0
		cubic_control = None
		quadratic_control = None
		if upper == "M":
			current_x = base_x + scanner.read_number()
			current_y = base_y + scanner.read_number()
			start_x, start_y = current_x, current_y
			segments.append(("M", current_x, current_y))
		elif upper == "L":
			current_x = base_x + scanner.read_number()
			current_y = base_y + scanner.read_number()
			segments.append(("L", current_x, current_y))
		elif upper == "H":
			current_x = base_x + scanner.read_number()
			segments.append(("L", current_x, current_y))
		elif upper == "V":
			current_y = base_y + scanner.read_number()
			segments.append(("L", current_x, current_y))
		elif upper == "C":
			control1_x = base_x + scanner.read_number()
			control1_y = base_y + scanner.read_number()
			control2_x = base_x + scanner.read_number()
			control2_y = base_y + scanner.read_number()
			current_x = base_x + scanner.read_number()
			current_y = base_y + scanner.read_number()
			segments.append(("C", control1_x, control1_y, control2_x, control2_y, current_x, current_y))
			cubic_control = (control2_x, control2_y)
		elif upper == "S":
			if last_cubic_control is not None:
				control1_x = 2.0 * current_x - last_cubic_control[0]
				control1_y = 2.0 * current_y - last_cubic_control[1]
			else:
				control1_x, control1_y = current_x, current_y
			control2_x = base_x + scanner.read_number()
			control2_y = base_y + scanner.read_number()
			current_x = base_x + scanner.read_number()
			current_y = base_y + scanner.read_number()
			segments.append(("C", control1_x, control1_y, control2_x, control2_y, current_x, current_y))
			cubic_control = (control2_x, control2_y)
		elif upper == "Q":
			control_x = base_x + scanner.read_number()
			control_y = base_y + scanner.read_number()
			current_x = base_x + scanner.read_number()
			current_y = base_y + scanner.read_number()
			segments.append(("Q", control_x, control_y, current_x, current_y))
			quadratic_control = (control_x, control_y)
		elif upper == "T":
			if last_quadratic_control is not None:
				control_x = 2.0 * current_x - last_quadratic_control[0]
				control_y = 2.0 * current_y - last_quadratic_control[1]
			else:
				control_x, control_y = current_x, current_y
			current_x = base_x + scanner.read_number()
			current_y = base_y + scanner.read_number()
			segments.append(("Q", control_x, control_y, current_x, current_y))
			quadratic_control = (control_x, control_y)
		elif upper == "A":
			radius_x = abs(scanner.read_number())
			radius_y = abs(scanner.read_number())
			rotation = scanner.read_number()
			large_arc = scanner.read_flag()
			sweep = scanner.read_flag()
			current_x = base_x + scanner.read_number()
			current_y = base_y + scanner.read_number()
			segments.append(("A", radius_x, radius_y, rotation, large_arc, sweep, current_x, current_y))
		elif upper == "Z":
			segments.append(("Z",))
			current_x, current_y = start_x, start_y
		else:
			raise IconError(f"unknown path command {command!r}")
		last_cubic_control = cubic_control
		last_quadratic_control = quadratic_control
	return segments


def arc_center_parameters(start_x, start_y, radius_x, radius_y, rotation, large_arc, sweep, end_x, end_y):
	"""
	Converts an endpoint-parameterized arc to its center form (SVG 1.1 implementation notes F.6.5,
	including the out-of-range radius correction of F.6.6).

	@param float start_x   Arc start x.
	@param float start_y   Arc start y.
	@param float radius_x  Requested x radius.
	@param float radius_y  Requested y radius.
	@param float rotation  X-axis rotation in degrees.
	@param int   large_arc Large-arc flag.
	@param int   sweep     Sweep flag (1 = positive angle direction).
	@param float end_x     Arc end x.
	@param float end_y     Arc end y.
	@return tuple (center_x, center_y, radius_x, radius_y, cosine, sine, start_angle, sweep_angle), with
		the radii corrected and the angles in radians.
	"""
	radians = math.radians(rotation)
	cosine = math.cos(radians)
	sine = math.sin(radians)
	half_dx = (start_x - end_x) / 2.0
	half_dy = (start_y - end_y) / 2.0
	prime_x = cosine * half_dx + sine * half_dy
	prime_y = -sine * half_dx + cosine * half_dy
	radius_x = abs(radius_x)
	radius_y = abs(radius_y)
	correction = (prime_x * prime_x) / (radius_x * radius_x) + (prime_y * prime_y) / (radius_y * radius_y)
	if correction > 1.0:
		radius_x *= math.sqrt(correction)
		radius_y *= math.sqrt(correction)
	numerator = radius_x * radius_x * radius_y * radius_y - radius_x * radius_x * prime_y * prime_y - radius_y * radius_y * prime_x * prime_x
	denominator = radius_x * radius_x * prime_y * prime_y + radius_y * radius_y * prime_x * prime_x
	factor = math.sqrt(max(numerator / denominator, 0.0)) if denominator > 0 else 0.0
	if large_arc == sweep:
		factor = -factor
	center_prime_x = factor * radius_x * prime_y / radius_y
	center_prime_y = -factor * radius_y * prime_x / radius_x
	center_x = cosine * center_prime_x - sine * center_prime_y + (start_x + end_x) / 2.0
	center_y = sine * center_prime_x + cosine * center_prime_y + (start_y + end_y) / 2.0

	def angle(ux, uy, vx, vy):
		"""
		Returns the signed angle from vector u to vector v.

		@param float ux u's x.
		@param float uy u's y.
		@param float vx v's x.
		@param float vy v's y.
		@return float The angle in radians.
		"""
		return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)

	start_angle = angle(1.0, 0.0, (prime_x - center_prime_x) / radius_x, (prime_y - center_prime_y) / radius_y)
	sweep_angle = angle(
		(prime_x - center_prime_x) / radius_x,
		(prime_y - center_prime_y) / radius_y,
		(-prime_x - center_prime_x) / radius_x,
		(-prime_y - center_prime_y) / radius_y,
	)
	if not sweep and sweep_angle > 0:
		sweep_angle -= 2.0 * math.pi
	elif sweep and sweep_angle < 0:
		sweep_angle += 2.0 * math.pi
	return (center_x, center_y, radius_x, radius_y, cosine, sine, start_angle, sweep_angle)


def arcs_to_cubics(segments):
	"""
	Replaces every elliptical arc with cubic Béziers of at most 90 degrees each.  Arcs are kept out of
	the output because their center is ill-conditioned under rounding: a near-semicircle (Blender's
	round end caps) moves its center visibly when its radius or endpoints round to a few decimals,
	while a cubic's control points round harmlessly.  Converting before the transform is exact, since
	an affine map takes a cubic to a cubic.  This is also what the editor's PathParser does with an arc.

	@param list segments Absolute segments that may hold arcs.
	@return list The same path with no arcs.
	"""
	converted = []
	current_x = current_y = 0.0
	start_x = start_y = 0.0
	for segment in segments:
		kind = segment[0]
		if kind == "A":
			radius_x, radius_y, rotation, large_arc, sweep, end_x, end_y = segment[1:]
			if (end_x, end_y) == (current_x, current_y):
				# SVG renders an arc onto its own start point as nothing.
				continue
			if radius_x == 0.0 or radius_y == 0.0:
				converted.append(("L", end_x, end_y))
			else:
				center_x, center_y, radius_x, radius_y, cosine, sine, start_angle, sweep_angle = arc_center_parameters(
					current_x, current_y, radius_x, radius_y, rotation, large_arc, sweep, end_x, end_y
				)
				pieces = max(1, math.ceil(abs(sweep_angle) / (math.pi / 2.0) - 1e-9))
				step_angle = sweep_angle / pieces
				handle = 4.0 / 3.0 * math.tan(step_angle / 4.0)

				def point_at(theta):
					"""
					Returns the ellipse point at a parametric angle.

					@param float theta The angle in radians.
					@return tuple The (x, y) point.
					"""
					ellipse_x = radius_x * math.cos(theta)
					ellipse_y = radius_y * math.sin(theta)
					return (center_x + cosine * ellipse_x - sine * ellipse_y, center_y + sine * ellipse_x + cosine * ellipse_y)

				def tangent_at(theta):
					"""
					Returns the ellipse's derivative with respect to its parametric angle.

					@param float theta The angle in radians.
					@return tuple The (dx, dy) tangent.
					"""
					ellipse_x = -radius_x * math.sin(theta)
					ellipse_y = radius_y * math.cos(theta)
					return (cosine * ellipse_x - sine * ellipse_y, sine * ellipse_x + cosine * ellipse_y)

				piece_start = (current_x, current_y)
				for piece in range(pieces):
					theta_start = start_angle + step_angle * piece
					theta_end = theta_start + step_angle
					piece_end = (end_x, end_y) if piece == pieces - 1 else point_at(theta_end)
					tangent_start = tangent_at(theta_start)
					tangent_end = tangent_at(theta_end)
					converted.append((
						"C",
						piece_start[0] + handle * tangent_start[0],
						piece_start[1] + handle * tangent_start[1],
						piece_end[0] - handle * tangent_end[0],
						piece_end[1] - handle * tangent_end[1],
						piece_end[0],
						piece_end[1],
					))
					piece_start = piece_end
			current_x, current_y = end_x, end_y
			continue
		converted.append(segment)
		if kind == "M":
			current_x, current_y = segment[1], segment[2]
			start_x, start_y = current_x, current_y
		elif kind == "Z":
			current_x, current_y = start_x, start_y
		else:
			current_x, current_y = segment[-2], segment[-1]
	return converted


def transform_segments(segments, matrix):
	"""
	Maps arc-free absolute segments through an affine matrix (see arcs_to_cubics).

	@param list  segments The absolute segments.
	@param tuple matrix   The transform.
	@return list The mapped segments.
	"""
	mapped = []
	for segment in segments:
		kind = segment[0]
		if kind in ("M", "L"):
			mapped.append((kind, *apply_point(matrix, segment[1], segment[2])))
		elif kind == "C":
			mapped.append((
				"C",
				*apply_point(matrix, segment[1], segment[2]),
				*apply_point(matrix, segment[3], segment[4]),
				*apply_point(matrix, segment[5], segment[6]),
			))
		elif kind == "Q":
			mapped.append(("Q", *apply_point(matrix, segment[1], segment[2]), *apply_point(matrix, segment[3], segment[4])))
		elif kind == "Z":
			mapped.append(segment)
		else:
			raise IconError(f"unexpected {kind} segment after arc conversion")
	return mapped


def format_number(value, precision):
	"""
	Formats a coordinate at fixed precision with trailing zeros trimmed.

	@param float value     The number.
	@param int   precision Decimal places kept.
	@return str The formatted number ("0" never carries a sign).
	"""
	text = f"{value:.{precision}f}".rstrip("0").rstrip(".")
	if text in ("-0", ""):
		return "0"
	return text


def format_segments(segments, precision):
	"""
	Writes absolute segments in the style of the icons already in Icon.kt: absolute commands,
	comma-joined coordinate pairs, space-separated pairs.

	@param list segments  The absolute segments.
	@param int  precision Decimal places kept.
	@return str The path data.
	"""
	def pair(x, y):
		"""
		Formats one coordinate pair.

		@param float x The x coordinate.
		@param float y The y coordinate.
		@return str "x,y".
		"""
		return f"{format_number(x, precision)},{format_number(y, precision)}"

	parts = []
	for segment in segments:
		kind = segment[0]
		if kind in ("M", "L"):
			parts.append(kind + pair(segment[1], segment[2]))
		elif kind == "C":
			parts.append("C" + " ".join((pair(segment[1], segment[2]), pair(segment[3], segment[4]), pair(segment[5], segment[6]))))
		elif kind == "Q":
			parts.append("Q" + " ".join((pair(segment[1], segment[2]), pair(segment[3], segment[4]))))
		else:
			parts.append("Z")
	return "".join(parts)


# ---------------------------------------------------------------------------------------------------
# Bounds (for the overflow report).


def segment_points(segments, steps=24):
	"""
	Samples points along arc-free absolute segments, densely enough for a bounds estimate.

	@param list segments The absolute segments.
	@param int  steps    Samples per curved segment.
	@return list The sampled (x, y) points.
	"""
	points = []
	current_x = current_y = 0.0
	start_x = start_y = 0.0
	for segment in segments:
		kind = segment[0]
		if kind == "M":
			# A moveto draws nothing; a segment drawn from here adds its own start point.
			current_x, current_y = segment[1], segment[2]
			start_x, start_y = current_x, current_y
		elif kind == "L":
			points.append((current_x, current_y))
			current_x, current_y = segment[1], segment[2]
			points.append((current_x, current_y))
		elif kind == "C":
			for step in range(0, steps + 1):
				t = step / steps
				u = 1.0 - t
				points.append((
					u * u * u * current_x + 3 * u * u * t * segment[1] + 3 * u * t * t * segment[3] + t * t * t * segment[5],
					u * u * u * current_y + 3 * u * u * t * segment[2] + 3 * u * t * t * segment[4] + t * t * t * segment[6],
				))
			current_x, current_y = segment[5], segment[6]
		elif kind == "Q":
			for step in range(0, steps + 1):
				t = step / steps
				u = 1.0 - t
				points.append((
					u * u * current_x + 2 * u * t * segment[1] + t * t * segment[3],
					u * u * current_y + 2 * u * t * segment[2] + t * t * segment[4],
				))
			current_x, current_y = segment[3], segment[4]
		else:
			current_x, current_y = start_x, start_y
	return points


# ---------------------------------------------------------------------------------------------------
# Shapes to paths.


def length_attribute(element, name, default=0.0):
	"""
	Reads a unitless length attribute (a px suffix is tolerated; any other unit is refused).

	@param Element element The shape.
	@param str     name    The attribute name.
	@param float   default Value when absent.
	@return float The length.
	"""
	raw = element.get(name)
	if raw is None or raw.strip() == "":
		return default
	raw = raw.strip()
	if raw.endswith("px"):
		raw = raw[:-2]
	try:
		return float(raw)
	except ValueError as error:
		raise IconError(f"unsupported length {name}={element.get(name)!r}") from error


def shape_segments(element, tag):
	"""
	Builds absolute segments for a shape element per the SVG shape-to-path equivalences.

	@param Element element The shape.
	@param str     tag     The local tag name.
	@return list|None The segments, or None for a shape that renders nothing (zero size).
	"""
	if tag == "path":
		return parse_path(element.get("d", ""))
	if tag == "rect":
		x = length_attribute(element, "x")
		y = length_attribute(element, "y")
		width = length_attribute(element, "width")
		height = length_attribute(element, "height")
		if width <= 0 or height <= 0:
			return None
		radius_x = element.get("rx")
		radius_y = element.get("ry")
		radius_x = length_attribute(element, "rx") if radius_x is not None else None
		radius_y = length_attribute(element, "ry") if radius_y is not None else None
		if radius_x is None:
			radius_x = radius_y if radius_y is not None else 0.0
		if radius_y is None:
			radius_y = radius_x
		radius_x = min(max(radius_x, 0.0), width / 2.0)
		radius_y = min(max(radius_y, 0.0), height / 2.0)
		if radius_x == 0 or radius_y == 0:
			return [("M", x, y), ("L", x + width, y), ("L", x + width, y + height), ("L", x, y + height), ("Z",)]
		corner = (radius_x, radius_y, 0.0, 0, 1)
		return [
			("M", x + radius_x, y),
			("L", x + width - radius_x, y),
			("A", *corner, x + width, y + radius_y),
			("L", x + width, y + height - radius_y),
			("A", *corner, x + width - radius_x, y + height),
			("L", x + radius_x, y + height),
			("A", *corner, x, y + height - radius_y),
			("L", x, y + radius_y),
			("A", *corner, x + radius_x, y),
			("Z",),
		]
	if tag in ("circle", "ellipse"):
		center_x = length_attribute(element, "cx")
		center_y = length_attribute(element, "cy")
		if tag == "circle":
			radius_x = radius_y = length_attribute(element, "r")
		else:
			radius_x = length_attribute(element, "rx")
			radius_y = length_attribute(element, "ry")
		if radius_x <= 0 or radius_y <= 0:
			return None
		quarter = (radius_x, radius_y, 0.0, 0, 1)
		return [
			("M", center_x + radius_x, center_y),
			("A", *quarter, center_x, center_y + radius_y),
			("A", *quarter, center_x - radius_x, center_y),
			("A", *quarter, center_x, center_y - radius_y),
			("A", *quarter, center_x + radius_x, center_y),
			("Z",),
		]
	if tag == "line":
		return [
			("M", length_attribute(element, "x1"), length_attribute(element, "y1")),
			("L", length_attribute(element, "x2"), length_attribute(element, "y2")),
		]
	# polyline / polygon
	values = [float(value) for value in NUMBER_PATTERN.findall(element.get("points", ""))]
	if len(values) < 4:
		return None
	segments = [("M", values[0], values[1])]
	for index in range(2, len(values) - 1, 2):
		segments.append(("L", values[index], values[index + 1]))
	if tag == "polygon":
		segments.append(("Z",))
	return segments


# ---------------------------------------------------------------------------------------------------
# Presentation properties.


def local_name(tag):
	"""
	Strips the namespace from an ElementTree tag.

	@param str tag The qualified tag.
	@return str The local name.
	"""
	return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def namespace_of(name):
	"""
	Returns the namespace URI of a qualified ElementTree name, or "" when it has none.

	@param str name The qualified tag or attribute name.
	@return str The namespace URI.
	"""
	return name[1:].split("}", 1)[0] if isinstance(name, str) and name.startswith("{") else ""


def parse_style(text):
	"""
	Splits a style attribute into an ordered property dict.

	@param str text The style attribute.
	@return dict Property name to value.
	"""
	properties = {}
	for declaration in (text or "").split(";"):
		if ":" in declaration:
			name, value = declaration.split(":", 1)
			properties[name.strip()] = value.strip()
	return properties


def format_style(properties):
	"""
	Joins a property dict back into a style attribute.

	@param dict properties Property name to value.
	@return str The style attribute.
	"""
	return ";".join(f"{name}:{value}" for name, value in properties.items())


def declared_property(element, name):
	"""
	Returns a property declared directly on an element (style wins over the presentation attribute),
	or None when it inherits.

	@param Element element The element.
	@param str     name    The property name.
	@return str|None The declared value.
	"""
	style = parse_style(element.get("style"))
	value = style.get(name, element.get(name))
	if value is None or value.strip() == "inherit":
		return None
	return value.strip()


def set_property(element, name, value):
	"""
	Writes a property where the element already declares it (style or attribute), else as an
	attribute; a None value removes it from both places.

	@param Element  element The element.
	@param str      name    The property name.
	@param str|None value   The new value.
	"""
	style = parse_style(element.get("style"))
	if name in style:
		if value is None:
			del style[name]
		else:
			style[name] = value
		if style:
			element.set("style", format_style(style))
		else:
			element.attrib.pop("style", None)
		if value is not None:
			element.attrib.pop(name, None)
			return
	if value is None:
		element.attrib.pop(name, None)
	else:
		element.set(name, value)


def leading_number(value):
	"""
	Reads the number at the start of a length value ("2", "2px", "2.5").

	@param str value The property value.
	@return float|None The number, or None when the value has none.
	"""
	match = NUMBER_PATTERN.match(value.strip())
	return float(match.group(0)) if match else None


# ---------------------------------------------------------------------------------------------------
# The bake.


class Baker:
	"""Bakes one SVG document in place and records what the bake had to approximate."""

	def __init__(self, root, root_matrix, precision):
		"""
		Prepares a bake.

		@param Element root        The parsed svg root.
		@param tuple   root_matrix Maps the source viewBox into the output box.
		@param int     precision   Decimal places kept in path data.
		"""
		self.root = root
		self.root_matrix = root_matrix
		self.precision = precision
		self.notes = []
		self.points = []
		self.gradient_references = []
		self.dropped = 0
		self.elements_by_id = {element.get("id"): element for element in root.iter() if element.get("id")}

	def run(self):
		"""
		Bakes the whole tree, then retargets gradients onto the baked coordinates.
		"""
		inherited = {"stroke": "none", "stroke-width": "1", "fill": "#000"}
		for name in ("stroke", "stroke-width", "fill"):
			declared = declared_property(self.root, name)
			if declared is not None:
				inherited[name] = declared
		self.walk_children(self.root, self.root_matrix, inherited)
		self.retarget_gradients()

	def walk_children(self, parent, matrix, inherited):
		"""
		Bakes each child of a container, dropping what does not render.

		@param Element parent    The container.
		@param tuple   matrix    The container's accumulated transform.
		@param dict    inherited The container's effective inherited properties.
		"""
		for child in list(parent):
			if not isinstance(child.tag, str):
				# Comments and processing instructions.
				parent.remove(child)
				continue
			if namespace_of(child.tag) in EDITOR_NAMESPACES:
				parent.remove(child)
				continue
			tag = local_name(child.tag)
			if namespace_of(child.tag) != SVG_NAMESPACE:
				raise IconError(f"unknown element {child.tag}")
			if tag in DROPPED_TAGS:
				parent.remove(child)
				continue
			if declared_property(child, "display") == "none":
				parent.remove(child)
				continue
			if tag in CONTAINER_TAGS or tag in SHAPE_TAGS:
				opacity = declared_property(child, "opacity")
				if opacity is not None and leading_number(opacity) == 0.0:
					# Blender keeps invisible spacer shapes in some icons; they draw nothing.
					self.dropped += 1
					parent.remove(child)
					continue
			strip_noise(child)
			if tag in PASSIVE_TAGS:
				if child.get("transform") is not None:
					raise IconError(f"<{tag}> carries a transform")
				for descendant in child.iter():
					strip_noise(descendant)
				for descendant in list(child):
					if namespace_of(descendant.tag) in EDITOR_NAMESPACES:
						child.remove(descendant)
				if tag == "defs" and len(child) == 0:
					parent.remove(child)
				continue
			child_matrix = matrix
			if child.get("transform") is not None:
				child_matrix = multiply(matrix, parse_transform(child.get("transform")))
				del child.attrib["transform"]
			child_inherited = dict(inherited)
			for name in INHERITED_PROPERTIES:
				declared = declared_property(child, name)
				if declared is not None:
					child_inherited[name] = declared
			if tag in CONTAINER_TAGS:
				# Stroke geometry is resolved on each shape, so the container's copy is dead weight.
				for name in ("stroke-width", "stroke-dasharray", "stroke-dashoffset"):
					set_property(child, name, None)
				self.walk_children(child, child_matrix, child_inherited)
				if len(child) == 0:
					parent.remove(child)
				elif tag == "g" and not child.attrib:
					# With its transform baked away the group carries nothing, so its children move up.
					index = list(parent).index(child)
					parent.remove(child)
					for offset, grandchild in enumerate(list(child)):
						parent.insert(index + offset, grandchild)
			elif tag in SHAPE_TAGS:
				if not self.bake_shape(child, tag, child_matrix, child_inherited):
					parent.remove(child)
			else:
				raise IconError(f"unsupported element <{tag}>")

	def bake_shape(self, element, tag, matrix, inherited):
		"""
		Turns one shape into a baked path in output coordinates.

		@param Element element   The shape (rewritten in place).
		@param str     tag       Its local tag name.
		@param tuple   matrix    Its accumulated transform.
		@param dict    inherited Its effective inherited properties.
		@return bool False when the shape renders nothing and should be dropped.
		"""
		segments = shape_segments(element, tag)
		if not segments:
			return False
		if not paints(inherited, "fill") and not paints(inherited, "stroke"):
			self.dropped += 1
			return False
		baked = transform_segments(arcs_to_cubics(segments), matrix)
		if tag != "path":
			element.tag = f"{{{SVG_NAMESPACE}}}path"
			for name in SHAPE_GEOMETRY_ATTRIBUTES:
				element.attrib.pop(name, None)
		element.set("d", format_segments(baked, self.precision))
		for name in ("{http://www.inkscape.org/namespaces/inkscape}original-d", "{http://www.inkscape.org/namespaces/inkscape}path-effect"):
			element.attrib.pop(name, None)

		major, minor = axis_scales(matrix)
		uniform_scale = math.sqrt(abs(determinant(matrix)))
		strokes = paints(inherited, "stroke")
		half_stroke = 0.0
		if strokes:
			width = leading_number(inherited.get("stroke-width", "1"))
			if width is None:
				raise IconError(f"unsupported stroke-width {inherited.get('stroke-width')!r}")
			set_property(element, "stroke-width", format_number(width * uniform_scale, self.precision + 2))
			half_stroke = width * major / 2.0
			for name in ("stroke-dasharray", "stroke-dashoffset"):
				value = inherited.get(name)
				if value is not None and value != "none":
					lengths = [format_number(float(number) * uniform_scale, self.precision + 2) for number in NUMBER_PATTERN.findall(value)]
					set_property(element, name, " ".join(lengths))
			if major - minor > 0.01 * major:
				self.notes.append(
					f"stroke on {element.get('id') or tag} under a non-uniform transform "
					f"(axis scales {major:.4g} vs {minor:.4g}); baked with the area-preserving width"
				)
		else:
			for name in ("stroke-width", "stroke-dasharray", "stroke-dashoffset", "stroke-linecap", "stroke-linejoin", "stroke-miterlimit"):
				set_property(element, name, None)

		for name in ("fill", "stroke"):
			match = URL_PATTERN.match(inherited.get(name, ""))
			if match:
				self.gradient_references.append((element, name, match.group(1), matrix))

		sampled = segment_points(baked)
		if half_stroke > 0:
			for x, y in sampled:
				self.points.append((x - half_stroke, y - half_stroke))
				self.points.append((x + half_stroke, y + half_stroke))
		else:
			self.points.extend(sampled)
		return True

	def retarget_gradients(self):
		"""
		Folds each referencing shape's baked transform into its gradient.  A userSpaceOnUse gradient
		lives in the shape's user space, which the bake replaced with output space, so the transform
		moves onto gradientTransform; a gradient shared by shapes with different transforms is cloned
		per transform.
		"""
		by_gradient = {}
		for element, name, gradient_id, matrix in self.gradient_references:
			by_gradient.setdefault(gradient_id, []).append((element, name, matrix))
		for gradient_id, references in by_gradient.items():
			gradient = self.elements_by_id.get(gradient_id)
			if gradient is None:
				raise IconError(f"missing gradient #{gradient_id}")
			if gradient.get(f"{{{XLINK_NAMESPACE}}}href") or gradient.get("href"):
				raise IconError(f"gradient #{gradient_id} inherits through href")
			units = gradient.get("gradientUnits", "objectBoundingBox")
			original_transform = parse_transform(gradient.get("gradientTransform", ""))
			assigned = {}
			for element, name, matrix in references:
				key = tuple(round(value, 9) for value in matrix)
				if key not in assigned:
					target = gradient
					if assigned:
						target = clone_element(gradient)
						target.set("id", f"{gradient_id}-{len(assigned) + 1}")
						parent = find_parent(self.root, gradient)
						parent.insert(list(parent).index(gradient) + 1, target)
					if units == "userSpaceOnUse":
						combined = multiply(matrix, original_transform)
						target.set("gradientTransform", "matrix(" + " ".join(format_number(value, 8) for value in combined) + ")")
					else:
						a, b, c, d, _, _ = matrix
						if abs(b) > 1e-9 or abs(c) > 1e-9 or a < 0 or d < 0:
							self.notes.append(f"objectBoundingBox gradient #{gradient_id} under a rotating or mirroring transform")
					assigned[key] = target.get("id")
				target_id = assigned[key]
				if target_id != gradient_id:
					declared = parse_style(element.get("style"))
					reference = f"url(#{target_id})"
					if name in declared:
						declared[name] = reference
						element.set("style", format_style(declared))
					else:
						element.set(name, reference)

	def bounds(self):
		"""
		Returns the baked geometry's bounding box in output coordinates (strokes included).

		@return tuple|None (min_x, min_y, max_x, max_y), or None when nothing renders.
		"""
		if not self.points:
			return None
		xs = [point[0] for point in self.points]
		ys = [point[1] for point in self.points]
		return (min(xs), min(ys), max(xs), max(ys))


def paints(inherited, name):
	"""
	Reports whether a shape's fill or stroke puts anything on screen.

	@param dict inherited The shape's effective inherited properties.
	@param str  name      "fill" or "stroke".
	@return bool False for a none / transparent paint, a zero paint opacity, or a zero stroke width.
	"""
	if inherited.get(name, "none").strip() in ("none", "transparent"):
		return False
	paint_opacity = inherited.get(f"{name}-opacity")
	if paint_opacity is not None and leading_number(paint_opacity) == 0.0:
		return False
	if name == "stroke" and leading_number(inherited.get("stroke-width", "1")) == 0.0:
		return False
	return True


def strip_noise(element):
	"""
	Drops editor metadata attributes and attributes with no rendering effect.

	@param Element element The element to clean.
	"""
	for name in list(element.attrib):
		if namespace_of(name) in EDITOR_NAMESPACES or name in NOISE_ATTRIBUTES:
			del element.attrib[name]
	style = parse_style(element.get("style"))
	vendor = [name for name in style if name.startswith("-inkscape-")]
	if vendor:
		for name in vendor:
			del style[name]
		if style:
			element.set("style", format_style(style))
		else:
			del element.attrib["style"]


def clone_element(element):
	"""
	Deep-copies an element.

	@param Element element The element.
	@return Element The copy.
	"""
	return ElementTree.fromstring(ElementTree.tostring(element))


def find_parent(root, target):
	"""
	Finds an element's parent.

	@param Element root   The tree root.
	@param Element target The child.
	@return Element The parent.
	"""
	for candidate in root.iter():
		if target in list(candidate):
			return candidate
	raise IconError("orphan element")


def source_box(root):
	"""
	Reads the source viewBox (falling back to width/height).

	@param Element root The svg root.
	@return tuple (min_x, min_y, width, height).
	"""
	view_box = root.get("viewBox")
	if view_box:
		values = [float(value) for value in NUMBER_PATTERN.findall(view_box)]
		if len(values) == 4 and values[2] > 0 and values[3] > 0:
			return tuple(values)
		raise IconError(f"bad viewBox {view_box!r}")
	width = length_attribute(root, "width")
	height = length_attribute(root, "height")
	if width <= 0 or height <= 0:
		raise IconError("no viewBox and no usable width/height")
	return (0.0, 0.0, width, height)


def placement_matrix(box, size, unit, fit, shrink=1.0):
	"""
	Builds the matrix that maps a source viewBox into the output box, centered.

	@param tuple box    The source (min_x, min_y, width, height).
	@param float size   The output box edge.
	@param float unit   Source units mapped onto the output edge under the unit fit.
	@param str   fit    "unit": size/unit for every icon, so icons keep Blender's relative scale;
	                    "box": each box's longer edge fills the output.
	@param float shrink Extra factor (at most 1) applied about the box center to pull overflowing art in.
	@return tuple The placement matrix.
	"""
	min_x, min_y, width, height = box
	scale = size / unit if fit == "unit" else size / max(width, height)
	scale *= shrink
	offset_x = (size - width * scale) / 2.0 - min_x * scale
	offset_y = (size - height * scale) / 2.0 - min_y * scale
	return (scale, 0.0, 0.0, scale, offset_x, offset_y)


def bake_tree(source_path, size, unit, fit, precision, shrink):
	"""
	Parses and bakes one icon file in memory.

	@param Path  source_path The source SVG.
	@param float size        The output box edge.
	@param float unit        Source units per output edge under the unit fit.
	@param str   fit         The placement rule (see placement_matrix).
	@param int   precision   Decimal places kept in path data.
	@param float shrink      Extra scale about the box center (see placement_matrix).
	@return tuple (tree, source box, baker, placement matrix).
	"""
	tree = ElementTree.parse(source_path)
	root = tree.getroot()
	if local_name(root.tag) != "svg":
		raise IconError("root is not <svg>")
	box = source_box(root)
	root_matrix = placement_matrix(box, size, unit, fit, shrink)
	if root.get("transform"):
		root_matrix = multiply(root_matrix, parse_transform(root.get("transform")))
	baker = Baker(root, root_matrix, precision)
	baker.run()
	return tree, box, baker, root_matrix


def overflow_shrink(bounds, size, tolerance):
	"""
	Returns the factor that pulls baked art inside the output box when scaled about its center.

	@param tuple|None bounds    The baked bounds (min_x, min_y, max_x, max_y).
	@param float      size      The output box edge.
	@param float      tolerance Overflow ignored as rounding (in output units).
	@return float The factor, 1 when the art already fits.
	"""
	if bounds is None:
		return 1.0
	center = size / 2.0
	reach = max(center - bounds[0], center - bounds[1], bounds[2] - center, bounds[3] - center)
	if reach <= center + tolerance:
		return 1.0
	return center / reach


def bake_file(source_path, destination_path, size, unit, fit, precision, overflow, tolerance):
	"""
	Bakes one icon file and writes it.

	@param Path  source_path      The source SVG.
	@param Path  destination_path Where the baked SVG goes.
	@param float size             The output box edge.
	@param float unit             Source units per output edge under the unit fit.
	@param str   fit              The placement rule (see placement_matrix).
	@param int   precision        Decimal places kept in path data.
	@param str   overflow         "shrink" pulls art that would leave the box back inside; "keep" lets it.
	@param float tolerance        Overflow ignored as rounding (in output units).
	@return tuple (source box, baked bounds, notes, placement matrix, shrink factor, invisible shapes dropped).
	"""
	tree, box, baker, root_matrix = bake_tree(source_path, size, unit, fit, precision, 1.0)
	shrink = overflow_shrink(baker.bounds(), size, tolerance) if overflow == "shrink" else 1.0
	if shrink < 1.0:
		tree, box, baker, root_matrix = bake_tree(source_path, size, unit, fit, precision, shrink)
	root = tree.getroot()

	strip_noise(root)
	for name in ("transform", "width", "height", "viewBox", "preserveAspectRatio", "x", "y"):
		root.attrib.pop(name, None)
	edge = format_number(size, 3)
	attributes = {"width": edge, "height": edge, "viewBox": f"0 0 {edge} {edge}"}
	attributes.update(root.attrib)
	root.attrib.clear()
	root.attrib.update(attributes)

	for element in root.iter():
		element.text = None
		element.tail = None
	ElementTree.indent(tree, space="\t")
	destination_path.parent.mkdir(parents=True, exist_ok=True)
	with open(destination_path, "w", encoding="utf-8") as output:
		output.write(ElementTree.tostring(root, encoding="unicode"))
		output.write("\n")
	return box, baker.bounds(), baker.notes, root_matrix, shrink, baker.dropped


def main():
	"""
	Bakes every SVG in the source directory and prints a report of approximations, shrunk icons, and
	any art left outside the box.

	@return int The process exit status (1 when any file failed).
	"""
	parser = argparse.ArgumentParser(description="Bake Blender icon SVGs into a centered square viewBox.")
	parser.add_argument("source", type=Path, help="directory of source SVGs")
	parser.add_argument("destination", type=Path, help="directory for baked SVGs (may equal the source to overwrite)")
	parser.add_argument("--size", type=float, default=24.0, help="output viewBox edge (default 24)")
	parser.add_argument("--unit", type=float, default=1600.0, help="source units mapped onto the output edge (default 1600, Blender's icon cell)")
	parser.add_argument(
		"--fit",
		choices=("unit", "box"),
		default="unit",
		help="unit: one scale for every icon, keeping Blender's relative sizes (default); box: each box's longer edge fills the output",
	)
	parser.add_argument(
		"--overflow",
		choices=("shrink", "keep"),
		default="shrink",
		help="shrink: art that would leave the box is scaled down about the box center until it fits (default); keep: leave it",
	)
	parser.add_argument("--precision", type=int, default=3, help="decimal places in path data (default 3)")
	parser.add_argument("--manifest", type=Path, help="also write each icon's placement matrix as JSON (for verification)")
	arguments = parser.parse_args()

	ElementTree.register_namespace("", SVG_NAMESPACE)
	ElementTree.register_namespace("xlink", XLINK_NAMESPACE)

	sources = sorted(arguments.source.glob("*.svg"))
	if not sources:
		print(f"no SVGs in {arguments.source}", file=sys.stderr)
		return 1
	failures = []
	placements = {}
	shrunk = []
	invisible = []
	overflows = []
	noted = []
	tolerance = 0.05
	for source_path in sources:
		destination_path = arguments.destination / source_path.name
		try:
			box, bounds, notes, placement, shrink, dropped = bake_file(
				source_path,
				destination_path,
				arguments.size,
				arguments.unit,
				arguments.fit,
				arguments.precision,
				arguments.overflow,
				tolerance,
			)
		except (IconError, ElementTree.ParseError, ValueError, ZeroDivisionError) as error:
			failures.append((source_path.name, str(error)))
			continue
		placements[source_path.name] = placement
		if dropped:
			invisible.append((source_path.name, dropped))
		if notes:
			noted.append((source_path.name, notes))
		if shrink < 1.0:
			shrunk.append((source_path.name, box, shrink))
		if bounds is not None:
			min_x, min_y, max_x, max_y = bounds
			if min_x < -tolerance or min_y < -tolerance or max_x > arguments.size + tolerance or max_y > arguments.size + tolerance:
				overflows.append((source_path.name, box, bounds))

	if arguments.manifest is not None:
		arguments.manifest.write_text(json.dumps({"size": arguments.size, "placements": placements}, indent="\t") + "\n")
	print(f"baked {len(sources) - len(failures)} of {len(sources)} icons into {arguments.destination}")
	if noted:
		print(f"\napproximated ({len(noted)}):")
		for name, notes in noted:
			for note in notes:
				print(f"  {name}: {note}")
	if invisible:
		print(f"\ndropped invisible shapes (opacity 0 or no paint) from {len(invisible)} icons:")
		print("  " + ", ".join(f"{name} ({count})" for name, count in invisible))
	if shrunk:
		print(f"\nshrunk to fit the box ({len(shrunk)}):")
		for name, box, shrink in shrunk:
			print(f"  {name}: source {format_number(box[2], 2)}x{format_number(box[3], 2)}, scaled by {shrink:.3f}")
	if overflows:
		print(f"\nart past the {format_number(arguments.size, 3)}-unit box ({len(overflows)}):")
		for name, box, bounds in overflows:
			print(
				f"  {name}: source {format_number(box[2], 2)}x{format_number(box[3], 2)}, "
				f"bounds x {bounds[0]:.2f}..{bounds[2]:.2f}, y {bounds[1]:.2f}..{bounds[3]:.2f}"
			)
	if failures:
		print(f"\nfailed ({len(failures)}):", file=sys.stderr)
		for name, message in failures:
			print(f"  {name}: {message}", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
