import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/core/widgets/inputs.dart';
import 'package:memory_verse/core/widgets/buttons.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/features/vaults/presentation/vault_detail_screen.dart';

class JoinVaultDialog extends ConsumerStatefulWidget {
  final String? initialCode;
  const JoinVaultDialog({super.key, this.initialCode});

  static void show(BuildContext context, {String? initialCode}) {
    showDialog(context: context, builder: (_) => JoinVaultDialog(initialCode: initialCode));
  }

  @override
  ConsumerState<JoinVaultDialog> createState() => _JoinVaultDialogState();
}

class _JoinVaultDialogState extends ConsumerState<JoinVaultDialog> {
  late final TextEditingController _codeController;
  bool _loading = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _codeController = TextEditingController(text: widget.initialCode ?? '');
  }

  @override
  void dispose() {
    _codeController.dispose();
    super.dispose();
  }

  String _extractCode(String input) {
    input = input.trim();
    // If it's a link: memoryverse://room/MV-8K4P2Q
    if (input.startsWith('memoryverse://room/')) {
      return input.replaceFirst('memoryverse://room/', '');
    }
    // If it's just the code
    return input;
  }

  Future<void> _submit() async {
    final input = _codeController.text.trim();
    if (input.isEmpty) {
      setState(() => _error = 'Please enter a code or link');
      return;
    }

    final code = _extractCode(input);

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final repo = ref.read(vaultRepositoryProvider);
      final room = await repo.joinVault(code);
      ref.invalidate(vaultsListProvider);

      if (mounted) {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Successfully joined room!'), backgroundColor: Colors.green),
        );
        Navigator.push(context, MaterialPageRoute(builder: (_) => VaultDetailScreen(vaultId: room.id)));
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _loading = false;
          _error = e.toString();
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;

    return Dialog(
      backgroundColor: c.surface,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(AppRadii.xl)),
      child: Padding(
        padding: const EdgeInsets.all(AppSpacing.s24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Join a Room',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: AppSpacing.s6),
            Text('Enter the room code or paste the invite link',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(color: c.textMuted)),
            const SizedBox(height: AppSpacing.s20),

            AppTextField(
              label: 'Room Code or Link',
              hint: 'e.g. MV-8K4P2Q or memoryverse://...',
              controller: _codeController,
            ),

            if (_error != null) ...[
              const SizedBox(height: AppSpacing.s12),
              Text(_error!, style: Theme.of(context).textTheme.bodySmall?.copyWith(color: c.error)),
            ],

            const SizedBox(height: AppSpacing.s24),
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: Text('Cancel', style: TextStyle(color: c.textMuted)),
                ),
                const SizedBox(width: AppSpacing.s12),
                SizedBox(
                  width: 120,
                  child: PrimaryButton(
                    label: 'Join',
                    onPressed: _loading ? null : _submit,
                    isLoading: _loading,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
