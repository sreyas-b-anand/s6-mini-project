import Navbar from "@/components/Navbar";
import { Poppins } from "next/font/google";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

export const metadata = {
  title: "RevDet - Bert Model",
  description: "A website for detecting fake reviews using machine learning",
};
const buttonProps = {
  href: "ml",
  label: "Try ML Model",
};
export default function BertLayout({ children }) {
  return (
    <div className="m-3">
      <Navbar buttonProps={buttonProps} />
      {children}
    </div>
  );
}
