import 'package:flutter/material.dart';
import 'package:memory_verse/core/theme/app_design_tokens.dart';

class FlowButton extends StatefulWidget {
  final String text;
  final VoidCallback onPressed;
  final bool isDark;

  const FlowButton({
    super.key,
    required this.text,
    required this.onPressed,
    this.isDark = false,
  });

  @override
  State<FlowButton> createState() => _FlowButtonState();
}

class _FlowButtonState extends State<FlowButton> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  late Animation<double> _circleExpandAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 250),
      reverseDuration: const Duration(milliseconds: 300),
    );

    _scaleAnimation = Tween<double>(begin: 1.0, end: 0.97).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOut),
    );

    _circleExpandAnimation = Tween<double>(begin: 0.0, end: 300.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeIn),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _handleTap() async {
    _controller.forward();
    await Future.delayed(const Duration(milliseconds: 200));
    widget.onPressed();
    
    Future.delayed(const Duration(milliseconds: 500), () {
      if (mounted) {
        _controller.reverse();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final defaultTextColor = widget.isDark ? Colors.white : AppColors.plum800;
    final hoverTextColor = widget.isDark ? AppColors.plum800 : Colors.white;
    final circleColor = widget.isDark ? Colors.white : AppColors.plum800;
    final borderColor = widget.isDark ? Colors.white24 : AppColors.plum800.withOpacity(0.25);

    return GestureDetector(
      onTap: _handleTap,
      child: AnimatedBuilder(
        animation: _controller,
        builder: (context, child) {
          return Transform.scale(
            scale: _scaleAnimation.value,
            child: Container(
              height: 54,
              width: 220,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(AppRadius.md),
                border: Border.all(
                  color: _controller.value > 0.1 ? Colors.transparent : borderColor,
                  width: 1.5,
                ),
                color: Colors.transparent,
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(AppRadius.md),
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    // Expanding Circle
                    Positioned(
                      child: Transform.scale(
                        scale: _circleExpandAnimation.value,
                        child: Container(
                          width: 20,
                          height: 20,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: circleColor,
                          ),
                        ),
                      ),
                    ),

                    // Content Row
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          widget.text,
                          style: AppTextStyles.body.copyWith(
                            fontWeight: FontWeight.w600,
                            color: Color.lerp(
                              defaultTextColor,
                              hoverTextColor,
                              _controller.value,
                            ),
                          ),
                        ),
                        const SizedBox(width: AppSpacing.s8),
                        Icon(
                          Icons.arrow_forward_rounded,
                          size: 18,
                          color: Color.lerp(
                            defaultTextColor,
                            hoverTextColor,
                            _controller.value,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
