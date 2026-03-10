"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Loader } from "lucide-react";
import usePost from "@/hooks/usePost";
import BTResultComponent from "./BTResultComponent";

const BTForm = () => {
  const { postData, loading, data } = usePost("/bert_score");

  const [text, setText] = useState("");
  const [link, setLink] = useState("");
  const [useLink, setUseLink] = useState(false);

  const handleSubmit = async () => {
    const formData = new FormData();


    if (useLink) {
      formData.append("link", link);
    } else {
      formData.append("text", text);
    }

    await postData({text});
  };

  return (
    <div className="max-w-xl mx-auto space-y-8">
      <div className="text-center space-y-2">
        <p className="font-semibold text-3xl">BERT Review Prediction</p>
        <p className="text-muted text-md">
          Enter a review or a product link below to predict its authenticity
        </p>
      </div>

      <div className="space-y-6">

        <div className="flex justify-end">
          <Button
            onClick={() => setUseLink(!useLink)}
            className="text-sm text-gray-100 focus:outline-none  hover:cursor-pointer hover:opacity-90"
          >
            {useLink ? "Enter reviews manually" : "Use product link"}
          </Button>
        </div>

        
        {!useLink && (
          <div className="space-y-2">
            <Label className="text-base">Review Text</Label>
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Write the review here..."
              className="min-h-40 focus-visible:ring-1"
            />
          </div>
        )}

        {/* Product Link Input */}
        {useLink && (
          <div className="space-y-2">
            <Label className="text-base">Product Link</Label>
            <Input
              value={link}
              onChange={(e) => setLink(e.target.value)}
              placeholder="Paste the product URL here..."
              className="focus-visible:ring-1"
            />
          </div>
        )}

        <Button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full py-6 text-background hover:cursor-pointer hover:opacity-90"
        >
          {loading ? <Loader className="animate-spin" /> : "Predict Review using BERT"}
        </Button>
      </div>

      {<BTResultComponent result={data} />}
    </div>
  );
};

export default BTForm;