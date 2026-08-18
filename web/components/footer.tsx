import Link from "next/link";
import { BookOpen, Globe, Mail, ExternalLink } from "lucide-react";

const links = {
  Product: ["Features", "Timeline", "Vaults", "AI Search"],
  Company: ["About", "Blog", "Careers", "Press"],
  Legal: ["Privacy", "Terms", "Cookies"],
};

const socialIcons = [Globe, Mail, ExternalLink];

export function Footer() {
  return (
    <footer className="border-t border-border/40 bg-background/50 backdrop-blur-sm py-16 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-10 mb-12">
          {/* Brand */}
          <div className="col-span-2 sm:col-span-1">
            <Link href="/" className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-xl bg-primary/20 border border-primary/40 flex items-center justify-center">
                <BookOpen className="w-4 h-4 text-primary" />
              </div>
              <span className="font-bold text-foreground">
                Memory<span className="text-primary">Verse</span>
              </span>
            </Link>
            <p className="text-sm text-muted-foreground leading-relaxed max-w-[200px]">
              Your AI-powered personal memory vault.
            </p>
            <div className="flex items-center gap-3 mt-5">
              {socialIcons.map((Icon, i) => (
                <Link
                  key={i}
                  href="#"
                  className="w-8 h-8 rounded-lg border border-border/50 flex items-center justify-center text-muted-foreground hover:text-foreground hover:border-primary/40 transition-colors"
                >
                  <Icon className="w-4 h-4" />
                </Link>
              ))}
            </div>
          </div>

          {/* Links */}
          {Object.entries(links).map(([section, items]) => (
            <div key={section}>
              <h4 className="text-xs font-semibold text-foreground uppercase tracking-widest mb-4">
                {section}
              </h4>
              <ul className="space-y-2.5">
                {items.map((item) => (
                  <li key={item}>
                    <Link
                      href="#"
                      className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {item}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="border-t border-border/40 pt-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-muted-foreground">
            © {new Date().getFullYear()} MemoryVerse. All rights reserved.
          </p>
          <p className="text-xs text-muted-foreground">
            Made with ♥ to preserve every moment.
          </p>
        </div>
      </div>
    </footer>
  );
}
