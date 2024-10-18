import 'dart:math';

import 'package:flutter/material.dart';
import '../widgets/point_painter.dart'; // Import the PointPainter

class PlotPointsPage extends StatelessWidget {
  final List<Map<String, double>> points;
  final String result;

  const PlotPointsPage({Key? key, required this.points, required this.result})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    int randomNumber = Random().nextInt(4);
    return Scaffold(
      appBar: AppBar(title: const Text('Predicted Event Plot')),
      body: SingleChildScrollView(
        child: SizedBox(
          width: 1500,
          height: 1000,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Padding(
                padding: const EdgeInsets.all(12),
                child: Text(
                  result,
                  style: const TextStyle(
                    fontSize: 45,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              Image.asset("assets/output$randomNumber.png", height: 400),
            ],
          ),
        ),
      ),
    );
  }
}
