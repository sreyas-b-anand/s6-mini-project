"use client";

import { SquareArrowOutUpRight } from "lucide-react";
import Link from "next/link";
import React from "react";

const Navbar = ({ buttonProps }) => {
  return (
    <div className="w-full py-4 flex justify-between items-center px-6 rounded-md bg-card shadow-[0_8px_30px_rgba(0,0,0,0.6)]/30 border border-border">
      {" "}
      <p className="font-semibold text-2xl">RevDet</p>
      <div className="bg-foreground text-background py-3 px-4 rounded-md flex items-center justify-center gap-2 text-sm font-medium hover:opacity-90 ">
        <Link
          className="flex  flex-row gap-2 items-center justify-center"
          href={buttonProps.href}
        >
          {buttonProps.label}
          <SquareArrowOutUpRight size={18} />
        </Link>
      </div>
    </div>
  );
};

export default Navbar;
