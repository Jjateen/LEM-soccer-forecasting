import 'package:flutter/material.dart';

class PointPainter extends CustomPainter {
  final List<Map<String, double>> points;

  PointPainter(this.points);

  @override
  void paint(Canvas canvas, Size size) {
    // Paint for the black border of the circles
    final borderPaint = Paint()
      ..color = Colors.black
      ..strokeWidth = 1
      ..style = PaintingStyle.stroke;

    // Loop through each point and scale it to the canvas size
    for (int i = 0; i < points.length; i++) {
      var point = points[i];

      // Calculate the color based on the point's index
      double t =
          i / (points.length - 1); // 't' is a normalized value between 0 and 1
      Color fillColor = Color.lerp(Colors.yellow, Colors.red, t)!;

      final fillPaint = Paint()
        ..color = fillColor
        ..style = PaintingStyle.fill;

      double x = point['x']! * 300; // X coordinate scaled to canvas width
      double y = (1 - point['y']!) * 300; // Y coordinate scaled and inverted

      // Draw the filled circle with the interpolated color
      canvas.drawCircle(Offset(x, y), 4, fillPaint);

      // Draw the black border around the circle
      canvas.drawCircle(Offset(x, y), 4, borderPaint);
    }

    // Draw the bounding rectangle (0, 0 to 1, 1) with black stroke
    final rectPaint = Paint()
      ..color = Colors.black
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), rectPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true; // Repaint when new points are added
  }
}
