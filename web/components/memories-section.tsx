"use client";

import { motion } from "motion/react";
import { Heart, MapPin, Calendar } from "lucide-react";

const memories = [
  {
    image: "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?w=600&q=80",
    title: "Road Trip Through Patagonia",
    date: "March 2024",
    location: "Argentina",
    category: "Travel",
    color: "from-orange-500/20 to-rose-500/20",
  },
  {
    image: "https://images.unsplash.com/photo-1530103862676-de8c9debad1d?w=600&q=80",
    title: "Dad's 60th Birthday",
    date: "November 2023",
    location: "Home",
    category: "Family",
    color: "from-purple-500/20 to-pink-500/20",
  },
  {
    image: "https://images.unsplash.com/photo-1523580494863-6f3031224c94?w=600&q=80",
    title: "Graduation Day",
    date: "June 2023",
    location: "University",
    category: "Milestone",
    color: "from-blue-500/20 to-cyan-500/20",
  },
  {
    image: "https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1?w=600&q=80",
    title: "First Solo Trek",
    date: "August 2023",
    location: "Swiss Alps",
    category: "Adventure",
    color: "from-green-500/20 to-teal-500/20",
  },
];

export function MemoriesSection() {
  return (
    <section id="memories" className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-14">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <span className="text-sm font-medium text-primary uppercase tracking-widest">
              Your Memories
            </span>
            <h2 className="text-4xl sm:text-5xl font-bold text-foreground mt-3 mb-4 tracking-tight">
              A life, beautifully stored
            </h2>
            <p className="text-muted-foreground text-lg max-w-xl mx-auto">
              See how MemoryVerse transforms everyday moments into a personal
              legacy you can revisit anytime.
            </p>
          </motion.div>
        </div>

        {/* Memory cards grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {memories.map((memory, i) => (
            <motion.div
              key={memory.title}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              className="group relative rounded-2xl overflow-hidden border border-border/40 bg-card/40 backdrop-blur-sm hover:scale-[1.02] hover:border-primary/30 transition-all duration-300 cursor-pointer"
              style={{ height: i % 2 === 0 ? "340px" : "280px" }}
            >
              {/* Background image */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={memory.image}
                alt={memory.title}
                className="absolute inset-0 w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
              />

              {/* Gradient overlay */}
              <div className={`absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent`} />
              <div className={`absolute inset-0 bg-gradient-to-br ${memory.color} opacity-40`} />

              {/* Heart */}
              <button className="absolute top-4 right-4 w-8 h-8 rounded-full bg-black/30 backdrop-blur-sm flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-black/50">
                <Heart className="w-4 h-4 text-white" />
              </button>

              {/* Category badge */}
              <div className="absolute top-4 left-4">
                <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-white/15 backdrop-blur-md text-white border border-white/20">
                  {memory.category}
                </span>
              </div>

              {/* Info */}
              <div className="absolute bottom-0 left-0 right-0 p-4">
                <h3 className="text-sm font-semibold text-white leading-snug mb-2">
                  {memory.title}
                </h3>
                <div className="flex items-center gap-3 text-white/60 text-xs">
                  <span className="flex items-center gap-1">
                    <Calendar className="w-3 h-3" />
                    {memory.date}
                  </span>
                  <span className="flex items-center gap-1">
                    <MapPin className="w-3 h-3" />
                    {memory.location}
                  </span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
