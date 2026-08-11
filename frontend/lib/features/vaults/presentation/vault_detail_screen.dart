import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:memory_verse/contracts/models.dart';
import 'package:memory_verse/core/design/tokens.dart';
import 'package:memory_verse/core/providers/app_providers.dart';
import 'package:memory_verse/core/repositories/app_repositories.dart';
import 'package:memory_verse/core/widgets/states.dart';
import 'package:memory_verse/features/vaults/presentation/room_share_sheet.dart';
import 'package:memory_verse/features/vaults/presentation/manage_members_sheet.dart';

class VaultDetailScreen extends ConsumerStatefulWidget {
  final String vaultId;
  const VaultDetailScreen({super.key, required this.vaultId});

  @override
  ConsumerState<VaultDetailScreen> createState() => _VaultDetailScreenState();
}

class _VaultDetailScreenState extends ConsumerState<VaultDetailScreen> {
  VaultModel? _vault;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadVaultData();
  }

  Future<void> _loadVaultData() async {
    try {
      final vRepo = ref.read(vaultRepositoryProvider);
      final vault = await vRepo.fetchVaultDetails(widget.vaultId);
      
      // Temporary hack: in a real implementation we would fetch memories by room_id.
      // Assuming we have a way to fetch memories belonging to this vault.
      
      setState(() {
        _vault = vault;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  void _showShareSheet() {
    if (_vault == null) return;
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => RoomShareSheet(vault: _vault!),
    );
  }

  void _showManageMembers() {
    if (_vault == null) return;
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => ManageMembersSheet(vault: _vault!, onMembersUpdated: _loadVaultData),
    );
  }

  Future<void> _leaveRoom() async {
    try {
      final repo = ref.read(vaultRepositoryProvider);
      await repo.leaveVault(widget.vaultId);
      ref.invalidate(vaultsListProvider);
      if (mounted) {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Left room')),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error leaving room: $e'), backgroundColor: Colors.red),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;

    if (_loading) {
      return Scaffold(
        backgroundColor: c.bg,
        appBar: AppBar(backgroundColor: c.bg),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    if (_error != null || _vault == null) {
      return Scaffold(
        backgroundColor: c.bg,
        appBar: AppBar(backgroundColor: c.bg),
        body: ErrorState(message: _error ?? 'Room not found', onRetry: _loadVaultData),
      );
    }

    final vault = _vault!;

    return Scaffold(
      backgroundColor: c.bg,
      body: CustomScrollView(
        physics: const BouncingScrollPhysics(),
        slivers: [
          SliverAppBar(
            pinned: true,
            expandedHeight: 220.0,
            backgroundColor: c.bg,
            iconTheme: IconThemeData(color: c.text),
            flexibleSpace: FlexibleSpaceBar(
              title: Text(vault.name,
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700, color: c.text)),
              background: Container(
                color: c.surfaceElevated,
                child: Center(
                  child: Icon(Icons.people_alt_outlined, size: 64, color: c.primary.withValues(alpha: 0.3)),
                ),
              ),
            ),
            actions: [
              IconButton(
                icon: const Icon(Icons.share_outlined),
                onPressed: _showShareSheet,
              ),
              PopupMenuButton<String>(
                icon: Icon(Icons.more_vert, color: c.text),
                onSelected: (val) {
                  if (val == 'leave') _leaveRoom();
                  if (val == 'members') _showManageMembers();
                },
                itemBuilder: (_) => [
                  const PopupMenuItem(value: 'members', child: Text('Manage Members')),
                  const PopupMenuItem(value: 'leave', child: Text('Leave Room', style: TextStyle(color: Colors.red))),
                ],
              ),
            ],
          ),

          // Room Details
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(AppSpacing.s24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (vault.description != null && vault.description!.isNotEmpty) ...[
                    Text(vault.description!, style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: c.textMuted, fontSize: 16)),
                    const SizedBox(height: AppSpacing.s24),
                  ],
                  Row(
                    children: [
                      Icon(Icons.group_outlined, size: 20, color: c.primary),
                      const SizedBox(width: AppSpacing.s8),
                      Text('${vault.memberCount} Members', style: TextStyle(fontFamily: 'Inter', fontWeight: FontWeight.w600, color: c.text)),
                      const Spacer(),
                      TextButton(
                        onPressed: _showManageMembers,
                        child: Text('View All', style: TextStyle(color: c.primary)),
                      ),
                    ],
                  ),
                  const Divider(height: 32),
                  Text('Shared Memories', style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
                  const SizedBox(height: AppSpacing.s16),
                ],
              ),
            ),
          ),

          // Shared Memories Placeholder
          const SliverPadding(
            padding: EdgeInsets.symmetric(horizontal: AppSpacing.s24),
            sliver: SliverToBoxAdapter(
              child: Padding(
                padding: EdgeInsets.all(AppSpacing.s32),
                child: EmptyState(
                  imageAsset: 'assets/images/empty_gallery.png',
                  title: 'No memories yet.',
                  subtitle: 'Create a memory and add it to this room to collaborate.',
                ),
              ),
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          // TODO: Open Create Memory sheet with this vaultId pre-selected
        },
        backgroundColor: c.primary,
        child: Icon(Icons.add, color: c.primaryInverse),
      ),
    );
  }
}
