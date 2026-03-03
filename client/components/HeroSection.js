"use client";
import React from "react";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
const HeroSection = () => {
  const router = useRouter();
  return (
    <>
      <div className="hero-bg"></div>

      <div className="max-w-4xl text-center">
        <motion.h1
          initial={{ opacity: 0, y: -40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-4xl sm:text-6xl font-bold leading-tight mb-6"
        >
          Detect Deception.
          <br />
          <span className="text-primary">Restore Online Trust.</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.8 }}
          className="text-muted text-lg sm:text-xl mb-12 max-w-2xl mx-auto"
        >
          Our Fake Review Detection System analyzes user reviews using advanced
          machine learning techniques to identify deceptive content and protect
          businesses and consumers from fraudulent opinions.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.8 }}
          className="flex flex-col sm:flex-row gap-6 justify-center"
        >
          <button
            onClick={() => router.push("/ml")}
            className="px-8 py-4 rounded-xl font-semibold text-lg
                       bg-primary
                       hover:scale-105
                       transition-all duration-300
                       shadow-xl"
          >
            Try Classical ML
          </button>

          <button
            onClick={() => router.push("/bert")}
            className="px-8 py-4 rounded-xl font-semibold text-lg
                       bg-secondary
                       hover:scale-105
                       transition-all duration-300
                       shadow-xl"
          >
            Try BERT Model
          </button>
        </motion.div>
      </div>
    </>
  );
};

export default HeroSection;
