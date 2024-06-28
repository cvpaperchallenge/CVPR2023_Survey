'use client'

import { motion } from 'framer-motion';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

export default function Home() {
  return (
    <div className="bg-gradient-to-br from-[var(--teal-2)] from-20% via-[var(--cyan-3)] via-50% to-[var(--teal-5)] to-90% text-white">

      <div className="container mx-auto px-4">
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center py-20"
        >
          <h1 className="text-5xl font-bold mb-4">Explore Cutting-Edge Computer Vision Research</h1>
          <p className="text-xl text-[#b0f0e6] mb-8">Summaries of top papers from CVPR, ICCV, and more</p>
          <Input
            className="max-w-xl mx-auto mb-8"
            placeholder="Search papers by title, author, or conference..."
          />
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12"
        >
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-[#102828] p-6 rounded-lg hover:scale-105 transition-transform">
              <h3 className="text-xl font-semibold mb-2">Featured Paper Title</h3>
              <p className="text-sm text-[#b0f0e6] mb-4">Authors et al.</p>
              <p className="text-sm mb-4">CVPR 2023</p>
              <Button>Read Summary</Button>
            </div>
          ))}
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="text-center mb-12"
        >
          <div className="grid grid-cols-3 gap-4">
            <div>
              <h4 className="text-4xl font-bold text-[#00c9a7]">1000+</h4>
              <p>Papers Summarized</p>
            </div>
            <div>
              <h4 className="text-4xl font-bold text-[#00c9a7]">10+</h4>
              <p>Conferences Covered</p>
            </div>
            <div>
              <h4 className="text-4xl font-bold text-[#00c9a7]">20+</h4>
              <p>Research Areas</p>
            </div>
          </div>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="text-center pb-20"
        >
          <Button size="lg">Explore All Papers</Button>
        </motion.section>
      </div>

    </div>
  );
}