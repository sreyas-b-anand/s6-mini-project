import Navbar from "@/components/Navbar";
import { Poppins } from "next/font/google";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

export const metadata = {
  title: "RevDet - ML Prediction",
  description: "A website for detecting fake reviews using machine learning",
};

export default function MLLayout({ children }) {
  return (
    <div className="m-3">
        <Navbar />
      {children}
    </div>
  );
}
