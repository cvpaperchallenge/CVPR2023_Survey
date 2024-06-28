'use client'

import { motion } from 'framer-motion';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { HiCursorClick } from "react-icons/hi";

const featuredPapers = [
  {
    id: 705,
    title: "Visual Atoms: Pre-Training Vision Transformers With Sinusoidal Waves",
    authors: "Sora Takashima, Ryo Hayamizu, Nakamasa Inoue, Hirokatsu Kataoka, Rio Yokota",
    conference: "CVPR 2023",
  },
  {
    id: 1098,
    title: "Graph Representation for Order-Aware Visual Transformation",
    authors: "Yue Qiu, Yanjun Sun, Fumiya Matsuzawa, Kenji Iwata, Hirokatsu Kataoka",
    conference: "CVPR 2023",
  },
  {
    id: 3631,
    title: "Frequency-aware GAN for Adversarial Manipulation Generation",
    authors: "Peifei Zhu, Genki Osada, Hirokatsu Kataoka, Tsubasa Takahashi",
    conference: "ICCV 2023",
  },
  {
    id: 3663,
    title: "Pre-training Vision Transformers with Very Limited Synthesized Images",
    authors: "Ryo Nakamura, Hirokatsu Kataoka, Sora Takashima, Edgar Josafat Martinez Noriega, Rio Yokota, Nakamasa Inoue",
    conference: "ICCV 2023",
  },
  {
    id: 4371,
    title: "SegRCDB: Semantic Segmentation via Formula-Driven Supervised Learning",
    authors: "Risa Shinoda, Ryo Hayamizu, Kodai Nakashima, Nakamasa Inoue, Rio Yokota, Hirokatsu Kataoka",
    conference: "ICCV 2023",
  }
];

export default function Home() {
  return (
    <div className="
      h-full grow
      bg-gradient-to-br
      from-[var(--teal-5)] dark:from-[var(--teal-2)] from-20%
      via-[var(--cyan-6)] dark:via-[var(--cyan-3)] via-50%
      to-[var(--teal-7)] dark:to-[var(--teal-5)] to-90%
    ">
      <div className="container flex flex-col items-center mx-auto px-4">
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center py-20"
        >
          <h1 className="text-[40px] leading-[48px] font-bold mb-4 text-foreground drop-shadow-md shadow-background">Explore Cutting-Edge Computer Vision Research</h1>
          <p className="text-xl text-card-foreground mb-8">Summaries of top papers from CVPR, ICCV, and more</p>
          {/* <Input
            className="max-w-xl mx-auto mb-8"
            placeholder="Search papers by title, author, or conference..."
          /> */}
          <motion.div
            whileHover={{ scale: 1.1 }}
          >
            <Button size="lg" className='rounded-2xl'>
              Explore All Papers
              <HiCursorClick className="ml-2 h-5 w-5" />
            </Button>
          </motion.div>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="w-4/5 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {featuredPapers.map((featuredPaper) => (
            <div key={featuredPaper.id} className="bg-[var(--teal-a4)] dark:bg-[var(--teal-a3)] p-6 rounded-lg hover:scale-105 transition-transform">
              <h3 className="text-xl text-foreground font-semibold mb-2">{featuredPaper.title}</h3>
              <p className="text-sm text-muted-foreground mb-4">{featuredPaper.authors}</p>
              <p className="text-sm text-foreground mb-4">{featuredPaper.conference}</p>
              {/* <Button>Read Summary</Button> */}
            </div>
          ))}
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="text-center w-4/5 py-20"
        >
          <div className="flex flex-row justify-around gap-20">
            <div>
              <h4 className="text-4xl font-bold text-[var(--teal-10)]">4000+</h4>
              <p className='text-card-foreground'>Papers Summarized</p>
            </div>
            <div>
              <h4 className="text-4xl font-bold text-[var(--teal-10)]">3</h4>
              <p className='text-card-foreground'>Conferences Covered</p>
            </div>
            <div>
              <h4 className="text-4xl font-bold text-[var(--teal-10)]">20+</h4>
              <p className='text-card-foreground'>Research Areas</p>
            </div>
          </div>
        </motion.section>
      </div>

    </div>
  );
}