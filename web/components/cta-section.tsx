"use client";

import { motion } from "motion/react";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { FlowButton } from "@/components/ui/flow-button";

export function CTASection() {
  return (
    <section id="download" className="py-24 px-6">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className="relative rounded-3xl border border-primary/20 bg-gradient-to-br from-primary/10 via-card/60 to-accent/10 backdrop-blur-xl p-10 sm:p-16 text-center overflow-hidden"
        >
          {/* Background glow orbs */}
          <div className="absolute top-0 left-1/4 w-64 h-64 bg-primary/15 rounded-full blur-[80px] pointer-events-none" />
          <div className="absolute bottom-0 right-1/4 w-64 h-64 bg-accent/10 rounded-full blur-[80px] pointer-events-none" />

          <div className="relative z-10">
            <span className="text-sm font-medium text-primary uppercase tracking-widest">
              Get the app
            </span>
            <h2 className="text-4xl sm:text-5xl font-bold text-foreground mt-4 mb-5 tracking-tight">
              Start preserving
              <br />
              your story today
            </h2>
            <p className="text-muted-foreground text-lg max-w-lg mx-auto mb-10">
              Free to download. Available on iOS and Android. Your memories are
              waiting to be remembered.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-5">
              <Link href="#">
                <FlowButton text="Download on iOS" />
              </Link>
              <Link href="#">
                <FlowButton text="Get on Android" dark />
              </Link>
            </div>

            <p className="mt-8 text-sm text-muted-foreground">
              Or{" "}
              <Link
                href="/sign-up"
                className="text-primary hover:underline inline-flex items-center gap-1 font-medium"
              >
                use the web app
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
