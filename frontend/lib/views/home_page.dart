import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with TickerProviderStateMixin {
  late AnimationController _controller;
  late AnimationController _fadeController;

  @override
  void initState() {
    super.initState();

    // Controller for hero section animation
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    );
    _controller.forward();

    // Controller for fading in feature section
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1),
    );
    Future.delayed(const Duration(milliseconds: 500), () {
      _fadeController.forward();
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _fadeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    var screenWidth = MediaQuery.of(context).size.width;

    return Scaffold(
      body: Stack(
        children: [
          SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                // Hero Section
                Stack(
                  children: [
                    _buildHeroSection(context, screenWidth),
                    _buildHeroText(context, screenWidth),
                    _buildHeroButton(context, screenWidth),
                  ],
                ),
                const SizedBox(height: 50),

                // Features Section
                _buildFeaturesSection(screenWidth),

                const SizedBox(height: 50),

                // Testimonials Section
                _buildTestimonialsSection(screenWidth),

                const SizedBox(height: 50),

                // Footer Section
                _buildFooterSection(),
              ],
            ),
          ),

          // Top-right Login/Profile Button
          Positioned(
            top: 16,
            right: 16,
            child: TextButton(
              onPressed: () {
                // Handle Login/Profile navigation here
                Navigator.pushNamed(context, '/login');
              },
              child: const Text(
                'Login',
                style: TextStyle(color: Colors.white, fontSize: 16),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeroSection(BuildContext context, double screenWidth) {
    return Container(
      height: screenWidth > 800
          ? MediaQuery.of(context).size.height
          : MediaQuery.of(context).size.height * 0.5,
      width: double.infinity,
      decoration: const BoxDecoration(
        image: DecorationImage(
          image: AssetImage('assets/4807.jpg'),
          fit: BoxFit.cover,
        ),
      ),
    );
  }

  Widget _buildHeroText(BuildContext context, double screenWidth) {
    return Positioned(
      top: screenWidth > 800
          ? MediaQuery.of(context).size.height * 0.2
          : MediaQuery.of(context).size.height * 0.1,
      left: 16,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SlideTransition(
            position:
                Tween<Offset>(begin: const Offset(-1, 0), end: Offset.zero)
                    .animate(CurvedAnimation(
                        parent: _controller, curve: Curves.easeOut)),
            child: Text(
              'NexMatch',
              style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                    fontSize: 60,
                  ),
            ),
          ),
          SlideTransition(
            position:
                Tween<Offset>(begin: const Offset(-1, 0), end: Offset.zero)
                    .animate(CurvedAnimation(
                        parent: _controller, curve: Curves.easeOut)),
            child: Text(
              'Welcome to Your Sports Hub',
              style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeroButton(BuildContext context, double screenWidth) {
    return Positioned(
      top: screenWidth > 800
          ? MediaQuery.of(context).size.height * 0.7
          : MediaQuery.of(context).size.height * 0.5,
      left: screenWidth * 0.4,
      right: screenWidth * 0.4,
      child: FadeTransition(
        opacity: _controller,
        child: ElevatedButton(
          onPressed: () {
            // Handle Get Started action here
            Navigator.pushNamed(context, '/input');
          },
          style: ElevatedButton.styleFrom(
            padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
            backgroundColor: Colors.blueAccent,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(50),
            ),
            fixedSize: Size.fromHeight(MediaQuery.sizeOf(context).height * 0.1),
          ),
          child: const Text(
            'Get Started',
            style: TextStyle(fontSize: 20),
          ),
        ),
      ),
    );
  }

  Widget _buildFeaturesSection(double screenWidth) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0),
      child: FadeTransition(
        opacity: _fadeController,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Features',
              style: Theme.of(context).textTheme.headlineLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 20),
            LayoutBuilder(
              builder: (context, constraints) {
                return screenWidth > 800
                    ? const Row(
                        mainAxisAlignment: MainAxisAlignment.spaceAround,
                        children: [
                          FeatureCard(
                            title: 'Real-Time Stats',
                            icon: Icons.bar_chart,
                            description:
                                'Track live performance of players and teams in real-time.',
                          ),
                          FeatureCard(
                            title: 'Personalized Insights',
                            icon: Icons.person_outline,
                            description:
                                'Get tailored insights and predictions based on your preferences.',
                          ),
                          FeatureCard(
                            title: 'Accurate Predictions',
                            icon: Icons.timeline,
                            description:
                                'Receive accurate match predictions powered by AI.',
                          ),
                        ],
                      )
                    : const SingleChildScrollView(
                        scrollDirection: Axis.horizontal,
                        child: Row(
                          children: [
                            FeatureCard(
                              title: 'Real-Time Stats',
                              icon: Icons.bar_chart,
                              description:
                                  'Track live performance of players and teams in real-time.',
                            ),
                            FeatureCard(
                              title: 'Personalized Insights',
                              icon: Icons.person_outline,
                              description:
                                  'Get tailored insights and predictions based on your preferences.',
                            ),
                            FeatureCard(
                              title: 'Accurate Predictions',
                              icon: Icons.timeline,
                              description:
                                  'Receive accurate match predictions powered by AI.',
                            ),
                          ],
                        ),
                      );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTestimonialsSection(double screenWidth) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0),
      child: FadeTransition(
        opacity: _fadeController,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'What Our Users Say',
              style: Theme.of(context).textTheme.headlineLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 20),
            screenWidth > 800
                ? const Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      Expanded(
                        child: TestimonialCard(
                          name: 'John Doe',
                          feedback:
                              'This platform has completely changed how I follow sports! The insights are so accurate.',
                        ),
                      ),
                      Expanded(
                        child: TestimonialCard(
                          name: 'Jane Smith',
                          feedback:
                              'I love the personalized recommendations. It helps me stay updated on my favorite teams!',
                        ),
                      ),
                    ],
                  )
                : const Column(
                    children: [
                      TestimonialCard(
                        name: 'John Doe',
                        feedback:
                            'This platform has completely changed how I follow sports! The insights are so accurate.',
                      ),
                      TestimonialCard(
                        name: 'Jane Smith',
                        feedback:
                            'I love the personalized recommendations. It helps me stay updated on my favorite teams!',
                      ),
                    ],
                  ),
          ],
        ),
      ),
    );
  }

  Widget _buildFooterSection() {
    return Container(
      width: double.infinity,
      color: Colors.grey[900],
      padding: const EdgeInsets.all(16.0),
      child: Column(
        children: [
          const Text(
            'Follow Us',
            style: TextStyle(color: Colors.white, fontSize: 18),
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              IconButton(
                icon: SvgPicture.asset(
                  'facebook.svg',
                  height: 30,
                ),
                onPressed: () {},
              ),
              IconButton(
                icon: SvgPicture.asset(
                  'twitter.svg',
                  height: 30,
                ),
                onPressed: () {},
              ),
              IconButton(
                icon: SvgPicture.asset(
                  'instagram.svg',
                  height: 30,
                ),
                onPressed: () {},
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// Custom Widget for Feature Cards
class FeatureCard extends StatelessWidget {
  final String title;
  final IconData icon;
  final String description;

  const FeatureCard({
    Key? key,
    required this.title,
    required this.icon,
    required this.description,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      width:
          MediaQuery.sizeOf(context).width * 0.3, // Adjusted for responsiveness
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(12.0),
        color: Colors.grey[200],
        boxShadow: const [
          BoxShadow(
            color: Colors.black12,
            blurRadius: 10,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          Icon(icon, size: 40, color: Colors.blueAccent),
          const SizedBox(height: 10),
          Text(
            title,
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 10),
          Text(
            description,
            textAlign: TextAlign.center,
            style: const TextStyle(fontSize: 14),
          ),
        ],
      ),
    );
  }
}

// Custom Widget for Testimonials
class TestimonialCard extends StatelessWidget {
  final String name;
  final String feedback;

  const TestimonialCard({
    Key? key,
    required this.name,
    required this.feedback,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 12),
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(12.0),
        color: Colors.grey[200],
        boxShadow: const [
          BoxShadow(
            color: Colors.black12,
            blurRadius: 10,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            name,
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Text(feedback),
        ],
      ),
    );
  }
}
