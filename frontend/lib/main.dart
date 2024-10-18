import 'package:flutter/material.dart';
import 'package:player_prediction/core/theme.dart';
import 'package:player_prediction/views/home_page.dart';
import 'package:player_prediction/views/input_event_page.dart';
import 'package:player_prediction/views/login_page.dart';
// import 'package:player_prediction/views/random_event_runner.dart';
import 'package:player_prediction/views/sign_up_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NexMatch',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme(),
      initialRoute: '/',
      routes: {
        '/': (context) => const HomePage(),
        '/login': (context) => const LoginPage(),
        '/signup': (context) => const SignupPage(),
        '/input': (context) => const InputEventPage(),
        // '/random': (context) => const RandomEventRunner(),
      },
    );
  }
}
