"use client";
import { useEffect } from "react";
import { Moon, SquareArrowOutUpRight, Sun } from "lucide-react";
import Link from "next/link";
import React from "react";
import { Button } from "@/components/ui/button";

const Navbar = ({ buttonProps }) => {
  // useEffect(() => {
  //   const saved = localStorage.getItem("theme");
  //   if (saved === "dark") {
  //     document.documentElement.classList.add("dark");
  //   }
  // }, []);

  // const toggleTheme = () => {
  //   const isDark = document.documentElement.classList.toggle(".dark");
  //   localStorage.setItem("theme", isDark ? "dark" : "light");
  // };
  return (
    <div className="w-full py-4 flex justify-between items-center px-6 rounded-md bg-background shadow-md ">
      {" "}
      <p className="font-semibold text-2xl">RevDet</p>
      {/* <div>
        <Button className={"bg-transparent"} onClick={toggleTheme}>
          <Moon size={18} color="black" />
          <Sun size={18} color="white" />
        </Button>
      </div> */}
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
