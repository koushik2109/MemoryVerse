import 'package:memory_verse/core/widgets/inputs.dart';
import 'package:memory_verse/core/widgets/buttons.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';



class CreateVaultDialog extends ConsumerStatefulWidget {
  const CreateVaultDialog({super.key});

  static void show(BuildContext context) {
    showDialog(context: context, builder: (_) => const CreateVaultDialog());
  }

  @override
  ConsumerState<CreateVaultDialog> createState() => _CreateVaultDialogState();
}

class _CreateVaultDialogState extends ConsumerState<CreateVaultDialog> {
  final _nameController = TextEditingController();
  final _descController = TextEditingController();
  bool _loading = false;
  String? _error;

  @override
  void dispose() {
    _nameController.dispose();
    _descController.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    final name = _nameController.text.trim();
    if (name.isEmpty) {
      setState(() => _error = 'Vault name is required');
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final repo = ref.read(vaultRepositoryProvider);
      await repo.createVault(
        name: name,
        description: _descController.text.trim().isEmpty ? null : _descController.text.trim(),
      );
      ref.invalidate(vaultsListProvider);
      ref.invalidate(userProfileProvider);

      if (mounted) {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Room created successfully!'), backgroundColor: Colors.green),
        );
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
            Text('Create New Room',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: AppSpacing.s6),
            Text('Organize your photos & videos with collaborators',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(color: c.textMuted)),
            const SizedBox(height: AppSpacing.s20),

            AppTextField(
              label: 'Room Name',
              hint: 'e.g. Summer Vacation 2026',
              controller: _nameController,
            ),
            const SizedBox(height: AppSpacing.s16),
            AppTextField(
              label: 'Description (Optional)',
              hint: 'e.g. Trips with friends to Goa',
              controller: _descController,
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
                    label: 'Create',
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
