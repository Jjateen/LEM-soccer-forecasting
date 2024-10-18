import 'package:flutter/material.dart';

class AppTheme {
  static const Color primaryBlue = Color(0xFF2563EB);   // Main brand color
  static const Color darkGray = Color(0xFF1F2937);      // Text and UI elements
  static const Color accentGreen = Color(0xFF10B981);   // Success, growth
  static const Color highlightYellow = Color(0xFFFBBF24); // Key features
  static const Color white = Color(0xFFFFFFFF);         // Backgrounds

  static ThemeData lightTheme() {
    return ThemeData(
      primaryColor: primaryBlue,
      scaffoldBackgroundColor: white,
      appBarTheme: const AppBarTheme(
        color: primaryBlue,
        iconTheme: IconThemeData(color: white),
        titleTextStyle: TextStyle(
          color: white,
          fontSize: 20,
          fontWeight: FontWeight.bold,
        ),
      ),
      textTheme: const TextTheme(
        headlineLarge: TextStyle(
          color: darkGray,
          fontSize: 24,
          fontWeight: FontWeight.bold,
        ),
        bodyLarge: TextStyle(
          color: darkGray,
          fontSize: 16,
        ),
      ),
      buttonTheme: const ButtonThemeData(
        buttonColor: primaryBlue,
        textTheme: ButtonTextTheme.primary,
      ),
      floatingActionButtonTheme: const FloatingActionButtonThemeData(
        backgroundColor: accentGreen,
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryBlue, // Blue buttons
          foregroundColor: white,     // White text on buttons
        ),
      ),
      colorScheme: const ColorScheme.light(
        primary: primaryBlue,
        secondary: accentGreen,
        onSurface: darkGray,
        error: Colors.red,
        surface: white,
      ),
      highlightColor: highlightYellow,
    );
  }
}
