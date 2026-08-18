import { ShaderBackground } from "@/components/ui/mesh-gradient";
import CarouselStacked from "@/components/ui/carousel-07";
import { Navbar } from "@/components/navbar";
import { HeroSection } from "@/components/hero-section";
import { FeaturesSection } from "@/components/features-section";
import { MemoriesSection } from "@/components/memories-section";
import { CTASection } from "@/components/cta-section";
import { Footer } from "@/components/footer";

export default function Home() {
  return (
    <main className="relative min-h-screen bg-background text-foreground overflow-x-hidden">
      {/* Mesh gradient background */}
      <div className="fixed inset-0 z-0 opacity-30 pointer-events-none">
        <ShaderBackground className="w-full h-full" />
      </div>

      {/* Dark overlay so text stays readable */}
      <div className="fixed inset-0 z-[1] pointer-events-none bg-gradient-to-b from-background/60 via-background/40 to-background/80" />

      {/* Content */}
      <div className="relative z-10">
        <Navbar />
        <HeroSection />

        {/* Recent Memories carousel */}
        <section className="py-16 px-4">
          <div className="max-w-6xl mx-auto text-center mb-8">
            <span className="text-xs font-semibold uppercase tracking-widest text-primary">
              Recent Memories
            </span>
            <h2 className="text-3xl sm:text-4xl font-bold text-foreground mt-3 mb-3">
              Moments worth reliving
            </h2>
            <p className="text-muted-foreground text-base max-w-xl mx-auto">
              Drag to explore your most recent captures — every swipe unlocks a story.
            </p>
          </div>
          <CarouselStacked />
        </section>

        <FeaturesSection />
        <MemoriesSection />
        <CTASection />
        <Footer />
      </div>
    </main>
  );
}
