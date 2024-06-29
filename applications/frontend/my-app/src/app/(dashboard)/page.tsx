'use client'

import { motion } from 'framer-motion'
import { HiCursorClick } from 'react-icons/hi'

import { Button } from '@/components/ui/button'

const featuredPapers = [
  {
    id: 705,
    title:
      'Visual Atoms: Pre-Training Vision Transformers With Sinusoidal Waves',
    authors:
      'Sora Takashima, Ryo Hayamizu, Nakamasa Inoue, Hirokatsu Kataoka, Rio Yokota',
    conference: 'CVPR 2023',
  },
  {
    id: 1098,
    title: 'Graph Representation for Order-Aware Visual Transformation',
    authors:
      'Yue Qiu, Yanjun Sun, Fumiya Matsuzawa, Kenji Iwata, Hirokatsu Kataoka',
    conference: 'CVPR 2023',
  },
  {
    id: 3631,
    title: 'Frequency-aware GAN for Adversarial Manipulation Generation',
    authors: 'Peifei Zhu, Genki Osada, Hirokatsu Kataoka, Tsubasa Takahashi',
    conference: 'ICCV 2023',
  },
  {
    id: 3663,
    title:
      'Pre-training Vision Transformers with Very Limited Synthesized Images',
    authors:
      'Ryo Nakamura, Hirokatsu Kataoka, Sora Takashima, Edgar Josafat Martinez Noriega, Rio Yokota, Nakamasa Inoue',
    conference: 'ICCV 2023',
  },
  {
    id: 4371,
    title:
      'SegRCDB: Semantic Segmentation via Formula-Driven Supervised Learning',
    authors:
      'Risa Shinoda, Ryo Hayamizu, Kodai Nakashima, Nakamasa Inoue, Rio Yokota, Hirokatsu Kataoka',
    conference: 'ICCV 2023',
  },
]

export default function Home() {
  return (
    <div
      className="
      h-full grow
      bg-gradient-to-br
      from-[var(--teal-5)] from-20% via-[var(--cyan-6)]
      via-50% to-[var(--teal-7)] to-90%
      dark:from-[var(--teal-2)] dark:via-[var(--cyan-3)] dark:to-[var(--teal-5)]
    "
    >
      <div className="container mx-auto flex flex-col items-center px-4">
        <motion.section
          animate={{ opacity: 1, y: 0 }}
          className="py-20 text-center"
          initial={{ opacity: 0, y: 20 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className="mb-4 text-[40px] font-bold leading-[48px] text-foreground shadow-background drop-shadow-md">
            Explore Cutting-Edge Computer Vision Research
          </h1>
          <p className="mb-8 text-xl text-card-foreground">
            Summaries of top papers from CVPR, ICCV, and more
          </p>
          {/* <Input
            className="max-w-xl mx-auto mb-8"
            placeholder="Search papers by title, author, or conference..."
          /> */}
          <motion.div whileHover={{ scale: 1.1 }}>
            <Button className="rounded-2xl" size="lg">
              Explore All Papers
              <HiCursorClick className="ml-2 size-5" />
            </Button>
          </motion.div>
        </motion.section>

        <motion.section
          animate={{ opacity: 1, y: 0 }}
          className="grid w-4/5 grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3"
          initial={{ opacity: 0, y: 20 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          {featuredPapers.map((featuredPaper) => (
            <div
              className="rounded-lg bg-[var(--teal-a4)] p-6 transition-transform hover:scale-105 dark:bg-[var(--teal-a3)]"
              key={featuredPaper.id}
            >
              <h3 className="mb-2 text-xl font-semibold text-foreground">
                {featuredPaper.title}
              </h3>
              <p className="mb-4 text-sm text-muted-foreground">
                {featuredPaper.authors}
              </p>
              <p className="mb-4 text-sm text-foreground">
                {featuredPaper.conference}
              </p>
              {/* <Button>Read Summary</Button> */}
            </div>
          ))}
        </motion.section>

        <motion.section
          animate={{ opacity: 1, y: 0 }}
          className="w-4/5 py-20 text-center"
          initial={{ opacity: 0, y: 20 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <div className="flex flex-row justify-around gap-20">
            <div>
              <h4 className="text-4xl font-bold text-[var(--teal-10)]">
                4000+
              </h4>
              <p className="text-card-foreground">Papers Summarized</p>
            </div>
            <div>
              <h4 className="text-4xl font-bold text-[var(--teal-10)]">3</h4>
              <p className="text-card-foreground">Conferences Covered</p>
            </div>
            <div>
              <h4 className="text-4xl font-bold text-[var(--teal-10)]">20+</h4>
              <p className="text-card-foreground">Research Areas</p>
            </div>
          </div>
        </motion.section>
      </div>
    </div>
  )
}
