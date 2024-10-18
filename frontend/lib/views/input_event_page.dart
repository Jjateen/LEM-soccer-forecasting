import 'package:flutter/material.dart';
import 'package:player_prediction/services/api_service.dart';
import 'package:player_prediction/utils/logger_util.dart';
import 'package:player_prediction/views/plot_points_page.dart';

class InputEventPage extends StatefulWidget {
  const InputEventPage({super.key});

  @override
  _InputEventPageState createState() => _InputEventPageState();
}

class _InputEventPageState extends State<InputEventPage> {
  String? eventType;
  bool isFirstHalf = true;
  double minute = 0.0;
  double xCoordinate = 0.5;
  double yCoordinate = 0.5;
  bool isHomeTeam = true;
  bool isAccurate = true;
  bool isGoal = false;
  int homeScore = 0;
  int awayScore = 0;

  final List<String> eventTypes = [
    'air_duel',
    'ground_attacking_duel',
    'ground_defending_duel',
    'ground_loose_ball_duel',
    'foul',
    'hand_foul',
    'late_card_foul',
    'out_of_game_foul',
    'protest',
    'simulation',
    'time_lost_foul',
    'violent_foul',
    'corner',
    'free_kick',
    'free_kick_cross',
    'goal_kick',
    'penalty',
    'throw_in',
    'goalkeeper_leaving_line',
    'acceleration',
    'clearance',
    'touch',
    'cross',
    'hand_pass',
    'head_pass',
    'high_pass',
    'launch',
    'simple_pass',
    'smart_pass',
    'reflexes',
    'save_attempt',
    'free_kick_shot',
    'shot'
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Input Match Event Data'),
      ),
      body: Container(
        width: double.infinity,
        decoration: const BoxDecoration(
            image: DecorationImage(
          image: AssetImage("stadium.jpg"),
          fit: BoxFit.fill,
        )),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: LayoutBuilder(
            builder: (context, constraints) {
              return constraints.maxWidth > 600
                  ? buildWideLayout()
                  : buildNarrowLayout();
            },
          ),
        ),
      ),
    );
  }

  Widget buildWideLayout() {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _buildDecoratedContainer(_buildLeftColumn()),
            ],
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildDecoratedContainer(_buildRightColumn()),
            ],
          ),
        ),
      ],
    );
  }

  Widget buildNarrowLayout() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildDecoratedContainer(
            [..._buildLeftColumn(), ..._buildRightColumn()]),
      ],
    );
  }

  List<Widget> _buildLeftColumn() {
    return [
      _buildDropdown(),
      const SizedBox(height: 24),
      _buildSwitchRow('First Half', isFirstHalf, (value) {
        setState(() => isFirstHalf = value);
      }),
      const SizedBox(height: 24),
      _buildCustomSlider('Minute (Normalized)', minute, (value) {
        setState(() => minute = value);
      }),
      const SizedBox(height: 24),
      _buildCustomSlider('X Coordinate (Normalized)', xCoordinate, (value) {
        setState(() => xCoordinate = value);
      }),
    ];
  }

  List<Widget> _buildRightColumn() {
    return [
      _buildCustomSlider('Y Coordinate (Normalized)', yCoordinate, (value) {
        setState(() => yCoordinate = value);
      }),
      const SizedBox(height: 24),
      _buildSwitchRow('Home Team', isHomeTeam, (value) {
        setState(() => isHomeTeam = value);
      }),
      const SizedBox(height: 24),
      _buildSwitchRow('Accurate', isAccurate, (value) {
        setState(() => isAccurate = value);
      }),
      const SizedBox(height: 24),
      _buildSwitchRow('Goal', isGoal, (value) {
        setState(() => isGoal = value);
      }),
      const SizedBox(height: 24),
      _buildScoreInputs(),
      const SizedBox(height: 32),
      Center(
        child: ElevatedButton(
          // onPressed: _handleSubmit,
          onPressed: _runMultiplePredictions,
          child: const Text('Submit Event Data'),
          style: ElevatedButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 40),
            textStyle: const TextStyle(fontSize: 18),
          ),
        ),
      ),
    ];
  }

  // Decorative Container with background color and rounded corners
  Widget _buildDecoratedContainer(List<Widget> children) {
    return Container(
      height: MediaQuery.sizeOf(context).height * 0.7,
      margin: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color.fromARGB(195, 238, 238, 238),
        borderRadius: BorderRadius.circular(12.0),
        border: Border.all(color: Colors.black, width: 1),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.3),
            spreadRadius: 3,
            blurRadius: 6,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      padding: const EdgeInsets.all(20.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: children,
      ),
    );
  }

  Widget _buildDropdown() {
    return DropdownButtonFormField<String>(
      decoration: const InputDecoration(
        labelText: 'Event Type',
        labelStyle: TextStyle(fontSize: 20),
        border: OutlineInputBorder(),
        contentPadding: EdgeInsets.all(20),
      ),
      value: eventType,
      onChanged: (String? newValue) {
        setState(() => eventType = newValue);
      },
      items: eventTypes.map((String type) {
        return DropdownMenuItem<String>(
          value: type,
          child: Text(type, style: const TextStyle(fontSize: 18)),
        );
      }).toList(),
    );
  }

  Widget _buildCustomSlider(
      String label, double value, ValueChanged<double> onChanged) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(fontSize: 18)),
        Slider(
          value: value,
          onChanged: onChanged,
          min: 0,
          max: 1,
          divisions: 100,
          label: value.toStringAsFixed(2),
        ),
      ],
    );
  }

  Widget _buildSwitchRow(
      String label, bool value, ValueChanged<bool> onChanged) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: const TextStyle(fontSize: 18)),
        Switch(value: value, onChanged: onChanged),
      ],
    );
  }

  Widget _buildScoreInputs() {
    return Row(
      children: [
        Expanded(
          child: _buildScoreInputField('Home Score', homeScore.toString(),
              (value) {
            setState(() => homeScore = int.tryParse(value) ?? 0);
          }),
        ),
        const SizedBox(width: 24),
        Expanded(
          child: _buildScoreInputField('Away Score', awayScore.toString(),
              (value) {
            setState(() => awayScore = int.tryParse(value) ?? 0);
          }),
        ),
      ],
    );
  }

  Widget _buildScoreInputField(
      String label, String initialValue, ValueChanged<String> onChanged) {
    return TextFormField(
      decoration: InputDecoration(
        labelText: label,
        labelStyle: const TextStyle(fontSize: 18),
        border: const OutlineInputBorder(),
        contentPadding: const EdgeInsets.all(20),
      ),
      keyboardType: TextInputType.number,
      initialValue: initialValue,
      onChanged: onChanged,
      style: const TextStyle(fontSize: 18),
    );
  }

  void _handleSubmit() async {
    Map<String, dynamic> data = {
      "event_type": eventTypes.indexOf(eventType!),
      "period": isFirstHalf ? 0 : 1,
      "minute": minute,
      "x": xCoordinate,
      "y": yCoordinate,
      "is_home_team": isHomeTeam,
      "home_score": homeScore,
      "away_score": awayScore,
    };

    String response = await ApiService("http://172.16.196.76:8000")
        .postRequest("predict", data);
    Log().debug(response);

    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Event Data Submitted',
              style: TextStyle(fontSize: 18)),
          content:
              Text(response.toString(), style: const TextStyle(fontSize: 16)),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('OK', style: TextStyle(fontSize: 16)),
            ),
          ],
        );
      },
    );
  }

  void _runMultiplePredictions() async {
    List<Map<String, double>> points = [];
    int homeS = 0, awayS = 0;

    Map<String, dynamic> data = {
      "event_type": eventTypes.indexOf(eventType!),
      "period": isFirstHalf ? 0 : 1,
      "minute": minute,
      "x": xCoordinate,
      "y": yCoordinate,
      "is_home_team": isHomeTeam,
      "home_score": homeScore,
      "away_score": awayScore,
    };
    String result = "";
    for (int i = 0; i < 1; i++) {
      var response = await ApiService("http://172.16.197.61:8000")
          .postRequest("predict", data);
      var prediction = response['prediction'];

      double predictedX = prediction['predicted_x'];
      double predictedY = prediction['predicted_y'];
      points.add({"x": predictedX, "y": predictedY});

      data['x'] = predictedX;
      data['y'] = predictedY;
      data['event_type'] = prediction['predicted_event_type'];
      result = prediction["match_result"];
    }

    // After 100 iterations, plot the points
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => PlotPointsPage(
          points: points,
          result: result,
        ),
      ),
    );
  }
}
