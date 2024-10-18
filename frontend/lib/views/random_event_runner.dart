// import 'package:flutter/material.dart';
// import '../services/dataset_service.dart';

// class RandomEventRunner extends StatefulWidget {
//   const RandomEventRunner({super.key});

//   @override
//   _RandomEventRunnerState createState() => _RandomEventRunnerState();
// }

// class _RandomEventRunnerState extends State<RandomEventRunner> {
//   List<dynamic>? selectedRow; // The selected random row from the dataset
//   String? result; // Placeholder for model result

//   // Column names for the dataset
//   final List<String> columnNames = [
//     'Event Type',
//     'Period',
//     'Minute',
//     'X Coordinate',
//     'Y Coordinate',
//     'Is Home Team',
//     'Is Accurate',
//     'Is Goal',
//     'Home Score',
//     'Away Score'
//   ];

//   @override
//   void initState() {
//     super.initState();
//     _loadRandomEvent(); // Load a random event on initialization
//   }

//   // Load a random row from a random dataset
//   Future<void> _loadRandomEvent() async {
//     List<dynamic> randomRow =
//         await DatasetService.getRandomEventFromRandomDataset();

//     // Ensure the row matches the number of columns (truncate if necessary)
//     if (randomRow.length > columnNames.length) {
//       randomRow =
//           randomRow.sublist(0, columnNames.length); // Truncate extra columns
//     }

//     // Don't pad missing columns, just ignore them

//     setState(() {
//       selectedRow = randomRow;
//     });
//   }

//   // Run the model on the selected row (placeholder for model integration)
//   void _runModel() {
//     if (selectedRow != null) {
//       setState(() {
//         result = 'Running model on: ${selectedRow.toString()}';
//         // Implement your model prediction logic here
//         // result = YourModel.run(selectedRow);
//       });
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         title: const Text('Run Model on Random Event'),
//       ),
//       body: Container(
//         width: MediaQuery.sizeOf(context).width,
//         decoration: const BoxDecoration(
//             gradient: RadialGradient(colors: [Colors.grey, Colors.blueAccent])),
//         child: Padding(
//           padding: const EdgeInsets.all(16.0),
//           child: Column(
//             mainAxisAlignment: MainAxisAlignment.center,
//             crossAxisAlignment: CrossAxisAlignment.start,
//             children: [
//               ElevatedButton(
//                 onPressed: _loadRandomEvent,
//                 child: const Text('Select Random Event'),
//               ),
//               const SizedBox(height: 20),
//               selectedRow != null
//                   ? Column(
//                       children: [
//                         const Text(
//                           'Selected Event:',
//                           style: TextStyle(
//                               fontSize: 18, fontWeight: FontWeight.bold),
//                         ),
//                         const SizedBox(height: 10),
//                         _buildEventTable(),
//                       ],
//                     )
//                   : const Text('No event selected yet.'),
//               const SizedBox(height: 20),
//               if (selectedRow != null)
//                 ElevatedButton(
//                   onPressed: _runModel,
//                   child: const Text('Run Model'),
//                 ),
//               const SizedBox(height: 20),
//               if (result != null)
//                 Text(
//                   'Model Output: $result',
//                   style: const TextStyle(fontWeight: FontWeight.bold),
//                 ),
//             ],
//           ),
//         ),
//       ),
//     );
//   }

//   // Build a table to display the selected row in a readable format
//   Widget _buildEventTable() {
//     return Table(
//       border: TableBorder.all(color: Colors.grey, width: 1),
//       defaultVerticalAlignment: TableCellVerticalAlignment.middle,
//       children: [
//         // Header row with column names
//         TableRow(
//           decoration: const BoxDecoration(color: Colors.grey),
//           children: columnNames.map((name) {
//             return Padding(
//               padding: const EdgeInsets.all(8.0),
//               child: Text(
//                 name,
//                 style: const TextStyle(fontWeight: FontWeight.bold),
//               ),
//             );
//           }).toList(),
//         ),
//         // Data row with the values from the selected row
//         TableRow(
//           children: selectedRow!.map((value) {
//             return Padding(
//               padding: const EdgeInsets.all(8.0),
//               child: Text(value.toString()),
//             );
//           }).toList(),
//         ),
//       ],
//     );
//   }
// }
