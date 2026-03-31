"use client";

import React from "react";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";

const HeroSection = () => {
  const router = useRouter();

  return (
    <section className="relative flex items-center justify-center min-h-[90vh] px-6 bg-background">
      <div className="hero-bg" />

      <div className="max-w-5xl text-center space-y-8">
        <motion.h1
          initial={{ opacity: 0, y: -40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-4xl sm:text-6xl lg:text-7xl font-bold leading-tight tracking-tight"
        >
          Detect Deception.
          <br />
          <span className="bg-linear-to-r from-indigo-500 via-blue-500 to-cyan-400 text-transparent bg-clip-text">
            Restore Online Trust.
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.8 }}
          className="text-muted text-base sm:text-lg max-w-2xl mx-auto leading-relaxed"
        >
          Identify fake reviews instantly using machine learning and NLP. Built
          to help users and businesses make smarter, safer decisions online.
        </motion.p>
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          <Button
            onClick={() => router.push("/ml")}
            className="px-8 py-6 text-base font-medium rounded-xl shadow-md hover:scale-105 transition-all"
          >
            Try Classical ML
          </Button>

          <Button
            variant="outline"
            onClick={() => router.push("/bert")}
            className="px-8 py-6 text-base font-medium rounded-xl hover:scale-105 transition-all hover:bg-foreground hover:text-background"
          >
            Try BERT Model
          </Button>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="text-xs text-muted pt-6"
        >
          Powered by Machine Learning • Real-time Analysis
        </motion.div>
      </div>
    </section>
  );
};

export default HeroSection;
