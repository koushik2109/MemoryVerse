import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';

class ManageMembersSheet extends ConsumerStatefulWidget {
  final VaultModel vault;
  final VoidCallback onMembersUpdated;
  const ManageMembersSheet({super.key, required this.vault, required this.onMembersUpdated});

  @override
  ConsumerState<ManageMembersSheet> createState() => _ManageMembersSheetState();
}

class _ManageMembersSheetState extends ConsumerState<ManageMembersSheet> {
  bool _loading = false;

  Future<void> _removeMember(String targetUserId) async {
    setState(() => _loading = true);
    try {
      final repo = ref.read(vaultRepositoryProvider);
      await repo.removeMember(widget.vault.id, targetUserId);
      widget.onMembersUpdated();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Member removed.')));
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e'), backgroundColor: Colors.red));
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final currentUser = ref.watch(userProfileProvider).valueOrNull;
    final isOwner = currentUser?.id == widget.vault.ownerId;

    return Container(
      decoration: BoxDecoration(
        color: c.surface,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(AppRadii.xl)),
      ),
      padding: const EdgeInsets.only(top: AppSpacing.s24, left: AppSpacing.s24, right: AppSpacing.s24),
      constraints: BoxConstraints(maxHeight: MediaQuery.of(context).size.height * 0.7),
      child: SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                margin: const EdgeInsets.only(bottom: AppSpacing.s24),
                decoration: BoxDecoration(
                  color: c.border,
                  borderRadius: BorderRadius.circular(AppRadii.sm),
                ),
              ),
            ),
            Text('Room Members', style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700, color: c.text)),
            const SizedBox(height: AppSpacing.s8),
            Text('${widget.vault.memberCount} people have joined this room.', style: TextStyle(color: c.textMuted)),
            const SizedBox(height: AppSpacing.s24),

            if (_loading) const LinearProgressIndicator(),

            Expanded(
              child: ListView.separated(
                itemCount: widget.vault.members.length,
                separatorBuilder: (_, __) => const Divider(height: 1),
                itemBuilder: (context, index) {
                  final member = widget.vault.members[index];
                  final isMe = member.userId == currentUser?.id;
                  
                  return ListTile(
                    contentPadding: EdgeInsets.zero,
                    leading: CircleAvatar(
                      backgroundColor: c.primary.withValues(alpha: 0.1),
                      backgroundImage: member.avatarUrl != null ? NetworkImage(member.avatarUrl!) : null,
                      child: member.avatarUrl == null
                          ? Text(member.fullName?.substring(0, 1).toUpperCase() ?? 'U', style: TextStyle(color: c.primary, fontWeight: FontWeight.bold))
                          : null,
                    ),
                    title: Text(
                      isMe ? 'You' : (member.fullName ?? 'Unknown User'),
                      style: TextStyle(fontWeight: FontWeight.w600, color: c.text),
                    ),
                    subtitle: Text(member.role.toUpperCase(), style: TextStyle(fontSize: 12, color: c.textMuted, letterSpacing: 0.5)),
                    trailing: isOwner && !isMe
                        ? IconButton(
                            icon: Icon(Icons.person_remove_outlined, color: c.error),
                            onPressed: () => _removeMember(member.userId),
                            tooltip: 'Remove from room',
                          )
                        : null,
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
