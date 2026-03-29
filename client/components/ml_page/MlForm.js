"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Loader } from "lucide-react";
import { motion } from "framer-motion";
import usePost from "@/hooks/usePost";
import MLResultComponent from "./MLResultComponent";

const MLForm = () => {
  const { postData, loading, data } = usePost("/ml_score");

  const [category, setCategory] = useState("");
  const [rating, setRating] = useState("");
  const [text, setText] = useState("");

  const handleSubmit = async () => {
    if (!category || !rating || !text) return;

    const formData = new FormData();
    formData.append("category", category);
    formData.append("rating", rating);
    formData.append("text", text);

    await postData(formData);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="max-w-2xl mx-auto p-8 surface card-hover space-y-8"
    >
      <div className="text-center space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight">
          Review Authenticity Checker
        </h1>
        <p className="text-muted text-sm">
          Detect whether a review is genuine using machine learning
        </p>
      </div>

      <div className="space-y-6">
        <div className="flex justify-between items-center gap-4 md:flex-nowrap flex-wrap">
          <div className="space-y-2 flex-3/5">
            <Label className="text-base">Category</Label>
            <Select
              onValueChange={setCategory}
              className="focus-visible:ring-1 "
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select category" />
              </SelectTrigger>

              <SelectContent className="bg-background border">
                <SelectItem value="Electronics">Electronics</SelectItem>
                <SelectItem value="Books">Books</SelectItem>
                <SelectItem value="Movies">Movies & TV</SelectItem>
                <SelectItem value="Home Appliances">Home & Kitchen</SelectItem>
                <SelectItem value="Sports">Sports & Outdoors</SelectItem>
                <SelectItem value="Tools and Home Improvements">
                  Tools & Home Improvement
                </SelectItem>
                <SelectItem value="Pets supplies">Pet Supplies</SelectItem>
                <SelectItem value="Kindle">Kindle Store</SelectItem>
                <SelectItem value="Toys">Toys & Games</SelectItem>
                <SelectItem value="Fashion and clothing">
                  Fashion & Clothing
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2 flex-2/5">
            <Label className="text-base">Rating</Label>
            <Input
              type="number"
              min="1"
              max="5"
              value={rating}
              onChange={(e) => setRating(e.target.value)}
              placeholder="Enter rating (1-5)"
              className="focus-visible:ring-1 "
            />
          </div>
        </div>

        <div className="space-y-2">
          <Label>Review</Label>
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Write a detailed review..."
            className="min-h-30 input-modern"
          />
        </div>

        <Button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full py-6 text-base font-medium button-primary glow hover:cursor-pointer"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <Loader className="animate-spin" size={18} />
              Analyzing
            </span>
          ) : (
            "Analyze Review"
          )}
        </Button>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: data ? 1 : 0, y: data ? 0 : 10 }}
        transition={{ duration: 0.4 }}
      >
        <MLResultComponent result={data} />
      </motion.div>
    </motion.div>
  );
};

export default MLForm;
