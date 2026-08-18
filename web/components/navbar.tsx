"use client";

import { useState } from "react";
import Link from "next/link";
import { Menu, X, BookOpen } from "lucide-react";

const navLinks = [
  { label: "Features", href: "#features" },
  { label: "Memories", href: "#memories" },
  { label: "Download", href: "#download" },
];

export function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 backdrop-blur-md border-b border-border/40 bg-background/30">
      {/* Logo */}
      <Link href="/" className="flex items-center gap-2 group">
        <div className="w-8 h-8 rounded-xl bg-primary/20 border border-primary/40 flex items-center justify-center group-hover:bg-primary/30 transition-colors">
          <BookOpen className="w-4 h-4 text-primary" />
        </div>
        <span className="font-bold text-lg tracking-tight text-foreground">
          Memory<span className="text-primary">Verse</span>
        </span>
      </Link>

      {/* Desktop links */}
      <div className="hidden md:flex items-center gap-8">
        {navLinks.map((l) => (
          <Link
            key={l.label}
            href={l.href}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            {l.label}
          </Link>
        ))}
      </div>

      {/* CTA */}
      <div className="hidden md:flex items-center gap-3">
        <Link
          href="/sign-in"
          className="text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          Sign in
        </Link>
        <Link
          href="/sign-up"
          className="text-sm px-4 py-2 rounded-xl bg-primary text-primary-foreground font-medium hover:bg-primary/90 transition-colors"
        >
          Get Started
        </Link>
      </div>

      {/* Mobile menu toggle */}
      <button
        className="md:hidden text-muted-foreground hover:text-foreground"
        onClick={() => setOpen(!open)}
        aria-label="Toggle menu"
      >
        {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
      </button>

      {/* Mobile menu */}
      {open && (
        <div className="absolute top-full left-0 right-0 bg-background/95 backdrop-blur-xl border-b border-border/40 py-4 px-6 flex flex-col gap-4 md:hidden">
          {navLinks.map((l) => (
            <Link
              key={l.label}
              href={l.href}
              onClick={() => setOpen(false)}
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              {l.label}
            </Link>
          ))}
          <div className="flex flex-col gap-2 pt-2 border-t border-border/40">
            <Link href="/sign-in" className="text-sm text-center py-2 text-muted-foreground hover:text-foreground">
              Sign in
            </Link>
            <Link href="/sign-up" className="text-sm text-center py-2 rounded-xl bg-primary text-primary-foreground font-medium">
              Get Started
            </Link>
          </div>
        </div>
      )}
    </nav>
  );
}
