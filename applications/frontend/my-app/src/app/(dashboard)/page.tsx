'use client'

import { motion } from 'framer-motion'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
import { HiCursorClick } from 'react-icons/hi'

import Metadata from '@/components/metadata'
import { Button } from '@/components/ui/button'

import { getPaperList } from '@/lib/fetch'

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
  useEffect(() => {
    void getPaperList()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const router = useRouter()
  return (
    <>
      <Metadata
        description={"Automated summaries of top conference papers using OpenAI's language model."}
        ogDescription={"Automated summaries of top conference papers using OpenAI's language model."}
        ogImage={`${process.env.NEXT_PUBLIC_FRONTEND_HOST_URL}/icon.png`}
        ogSiteName='LLM Survey'
        ogTitle={`Home`}
        ogType='website'
        ogUrl={`${process.env.NEXT_PUBLIC_FRONTEND_HOST_URL}/`}
        title={`Home | LLM Survey`}
        twitterCard='summary'
        twitterSite='@CVpaperChalleng'
      />
      <div
        // eslint-disable-next-line tailwindcss/no-contradicting-classname
        className="
        h-full grow
        bg-gradient-to-br
        from-[var(--teal-5)] from-20% via-[var(--cyan-6)]
        via-50% to-[var(--teal-7)] to-90%
        dark:from-[var(--teal-2)] dark:via-[var(--cyan-3)] dark:to-[var(--teal-5)]
      "
      >
        <div className="mx-auto flex min-w-[320px] max-w-[1200px] flex-col items-center px-4">
          <motion.section
            animate={{ opacity: 1, y: 0 }}
            className="pb-10 pt-16 text-center min-[461px]:px-5 min-[461px]:max-[600px]:pb-16 min-[461px]:max-[600px]:pt-20 min-[601px]:py-20"
            initial={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="mb-2 text-2xl font-bold leading-8 text-foreground shadow-background drop-shadow-md min-[461px]:max-[600px]:text-3xl min-[461px]:max-[600px]:leading-10 min-[601px]:mb-4 min-[601px]:text-[40px] min-[601px]:leading-[48px]">
              Explore Cutting-Edge Computer Vision Research
            </h1>
            <p className="mb-5 text-sm text-card-foreground min-[461px]:max-[600px]:text-base min-[601px]:mb-8 min-[601px]:text-xl">
              Summaries of top papers from CVPR, ICCV, and more
            </p>
            <motion.div whileHover={{ scale: 1.1 }}>
              <Button className="rounded-2xl" onClick={() => router.push('/list')} size="lg">
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
                className="cursor-pointer rounded-lg bg-[var(--teal-a4)] p-5 transition-transform hover:scale-105 dark:bg-[var(--teal-a3)] min-[461px]:p-6"
                key={featuredPaper.id}
                onClick={() => router.push(`/details?id=${featuredPaper.id}`)}
              >
                <h3 className="mb-2 text-base font-semibold text-foreground min-[461px]:max-[600px]:text-lg min-[601px]:text-xl">
                  {featuredPaper.title}
                </h3>
                <p className="mb-3 text-xs text-muted-foreground min-[461px]:mb-4 min-[601px]:text-sm">
                  {featuredPaper.authors}
                </p>
                <p className="mb-3 text-xs text-foreground min-[461px]:mb-4 min-[601px]:text-sm">
                  {featuredPaper.conference}
                </p>
              </div>
            ))}
          </motion.section>

          <motion.section
            animate={{ opacity: 1, y: 0 }}
            className="w-4/5 max-w-[730px] pb-16 pt-10 text-center min-[461px]:max-[600px]:pb-20 min-[461px]:max-[600px]:pt-16 min-[601px]:py-20"
            initial={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <div className="flex flex-row justify-between gap-4 min-[461px]:max-[600px]:gap-10 min-[601px]:gap-20">
              <div>
                <h4 className="text-xl font-bold text-[var(--teal-10)] min-[461px]:max-[600px]:text-2xl min-[601px]:text-4xl">
                  4000+
                </h4>
                <p className="text-xs text-card-foreground min-[461px]:max-[600px]:text-sm min-[601px]:text-base">
                  Papers Summarized
                </p>
              </div>
              <div>
                <h4 className="text-xl font-bold text-[var(--teal-10)] min-[461px]:max-[600px]:text-2xl min-[601px]:text-4xl">
                  3
                </h4>
                <p className="text-xs text-card-foreground min-[461px]:max-[600px]:text-sm min-[601px]:text-base">
                  Conferences Covered
                </p>
              </div>
              <div>
                <h4 className="text-xl font-bold text-[var(--teal-10)] min-[461px]:max-[600px]:text-2xl min-[601px]:text-4xl">
                  20+
                </h4>
                <p className="text-xs text-card-foreground min-[461px]:max-[600px]:text-sm min-[601px]:text-base">
                  Research Areas
                </p>
              </div>
            </div>
          </motion.section>
        </div>
      </div>
    </>
  )
}
