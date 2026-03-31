"use client";

import { motion } from "framer-motion";
import React from "react";

const Toggle = ({ useLink, setUseLink }) => {
  return (
    <div className="relative flex bg-secondary p-1 rounded-lg w-64 mx-auto">
      <motion.div
        layout
        transition={{ type: "spring", stiffness: 300, damping: 25 }}
        className="absolute top-1 bottom-1 w-1/2 bg-primary rounded-md"
        style={{
          left: useLink ? "50%" : "0%",
        }}
      />

      <button
        onClick={() => setUseLink(false)}
        className={`relative z-10 flex items-center justify-center w-1/2 py-1.5 text-sm rounded-md transition ${
          !useLink ? "text-white" : "text-foreground"
        }`}
      >
        Manual
      </button>

      <button
        onClick={() => setUseLink(true)}
        className={`relative z-10 flex items-center justify-center w-1/2 py-1.5 text-sm rounded-md transition ${
          useLink ? "text-white" : "text-foreground"
        }`}
      >
        Link
      </button>
    </div>
  );
};

export default Toggle;
