"use client";

import { motion } from "motion/react";
import Link from "next/link";
import { Sparkles } from "lucide-react";
import { FlowButton } from "@/components/ui/flow-button";

export function HeroSection() {
  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-6 pt-24 pb-16 text-center">
      {/* Glow blob */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="w-[600px] h-[600px] rounded-full bg-primary/8 blur-[140px]" />
      </div>

      <div className="relative z-10 max-w-4xl mx-auto">
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-primary/30 bg-primary/10 text-primary text-sm font-medium mb-8"
        >
          <Sparkles className="w-3.5 h-3.5" />
          AI-Powered Memory Vault
        </motion.div>

        {/* Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.1 }}
          className="text-5xl sm:text-6xl lg:text-7xl font-bold leading-[1.05] tracking-tight text-foreground mb-6"
        >
          Every moment
          <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-accent to-primary">
            lives forever.
          </span>
        </motion.h1>

        {/* Subheading */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}
          className="text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto mb-12 leading-relaxed"
        >
          MemoryVerse lets you capture, preserve, and relive your most precious
          memories — organised by AI, beautifully presented in personal vaults.
        </motion.p>

        {/* CTAs — FlowButton */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.3 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-5"
        >
          <Link href="/sign-up">
            <FlowButton text="Start for free" />
          </Link>
          <Link href="#features">
            <FlowButton text="See how it works" dark />
          </Link>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.6 }}
          className="flex flex-wrap items-center justify-center gap-10 mt-16"
        >
          {[
            { value: "10K+", label: "Memories saved" },
            { value: "98%", label: "Recall rate" },
            { value: "5★", label: "App store rating" },
          ].map((stat) => (
            <div key={stat.label} className="flex flex-col items-center">
              <span className="text-3xl font-bold text-foreground">{stat.value}</span>
              <span className="text-sm text-muted-foreground mt-1">{stat.label}</span>
            </div>
          ))}
        </motion.div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1"
      >
        <span className="text-xs text-muted-foreground">Scroll to explore</span>
        <motion.div
          animate={{ y: [0, 6, 0] }}
          transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
          className="w-0.5 h-8 bg-gradient-to-b from-primary/60 to-transparent rounded-full"
        />
      </motion.div>
    </section>
  );
}
