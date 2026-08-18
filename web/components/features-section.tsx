"use client";

import { motion } from "motion/react";
import {
  Brain,
  Clock,
  Lock,
  Share2,
  Search,
  Zap,
} from "lucide-react";

const features = [
  {
    icon: Brain,
    title: "AI-Powered Organisation",
    description:
      "Let AI automatically tag, categorise, and connect your memories — so you never lose a moment.",
  },
  {
    icon: Clock,
    title: "Smart Timeline",
    description:
      "Relive your story in a beautiful chronological timeline that surfaces the right memory at the right time.",
  },
  {
    icon: Lock,
    title: "Private Vaults",
    description:
      "Store your most personal memories in encrypted vaults — only you hold the key.",
  },
  {
    icon: Share2,
    title: "Shared Memories",
    description:
      "Invite friends and family to co-create memory collections and share experiences together.",
  },
  {
    icon: Search,
    title: "Semantic Search",
    description:
      "Search for 'that beach trip in 2023' or 'mom's birthday' — AI understands what you mean.",
  },
  {
    icon: Zap,
    title: "Instant Recall",
    description:
      "Surface old memories on anniversaries, similar days, or when you need a dose of nostalgia.",
  },
];

export function FeaturesSection() {
  return (
    <section id="features" className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <span className="text-sm font-medium text-primary uppercase tracking-widest">
              Features
            </span>
            <h2 className="text-4xl sm:text-5xl font-bold text-foreground mt-3 mb-4 tracking-tight">
              Built for remembering
            </h2>
            <p className="text-muted-foreground text-lg max-w-xl mx-auto">
              Everything you need to capture life as it happens and find it
              again years later.
            </p>
          </motion.div>
        </div>

        {/* Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, i) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.08 }}
              className="group relative p-6 rounded-2xl border border-border/50 bg-card/50 backdrop-blur-sm hover:border-primary/40 hover:bg-card/80 transition-all duration-300 overflow-hidden"
            >
              {/* Hover glow */}
              <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 bg-gradient-to-br from-primary/5 to-transparent pointer-events-none" />

              <div className="w-11 h-11 rounded-xl bg-primary/15 border border-primary/25 flex items-center justify-center mb-4 group-hover:bg-primary/25 transition-colors">
                <feature.icon className="w-5 h-5 text-primary" />
              </div>
              <h3 className="text-base font-semibold text-foreground mb-2">
                {feature.title}
              </h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
