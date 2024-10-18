// import 'dart:math';
// import 'package:csv/csv.dart';
// import 'package:flutter/services.dart';

// class DatasetService {
//   // List of dataset paths
//   static const List<String> datasetPaths = [
//     'England.csv',
//     'European_Championship.csv',
//     'France.csv',
//     'Germany.csv',
//     'Italy.csv',
//     'Spain.csv',
//     'World_Cup.csv',
//   ];

//   // Load a specific dataset from assets
//   static Future<List<List<dynamic>>> _loadDataset(String path) async {
//     final rawData = await rootBundle.loadString(path);
//     return const CsvToListConverter().convert(rawData);
//   }

//   // Select a random row from a randomly selected dataset
//   static Future<List<dynamic>> getRandomEventFromRandomDataset() async {
//     // Select a random dataset index
//     final randomDatasetIndex = Random().nextInt(datasetPaths.length);
//     final selectedDatasetPath = datasetPaths[randomDatasetIndex];

//     // Load only the selected dataset
//     final selectedDataset = await _loadDataset(selectedDatasetPath);

//     // Select a random row from the loaded dataset
//     final randomRowIndex = Random().nextInt(selectedDataset.length);
//     return selectedDataset[randomRowIndex];
//   }
// }
