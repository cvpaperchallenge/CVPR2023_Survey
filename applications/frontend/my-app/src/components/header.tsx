'use client'
import Image from 'next/image'
import { PiXLogo } from "react-icons/pi";
import { RxGithubLogo } from "react-icons/rx";
import { MdFeedback } from "react-icons/md";
import whiteLogo from '../../public/cc_logo_2white.png'
import blackLogo from '../../public/cc_logo_2.png'

import { Button } from '@/components/ui/button'
import {Separator} from '@radix-ui/react-separator'

export default function Header() {
  return (
    <div className="flex h-full items-center justify-between text-primary py-2.5 px-4">
      <>
        <Image
          src={whiteLogo}
          alt="logo"
          sizes="100vw"
          className="w-5/12 h-auto min-w-36 max-w-80 hidden dark:block"
        />
        <Image
          src={blackLogo}
          alt="logo"
          sizes="100vw"
          className="w-5/12 h-auto min-w-36 max-w-80 dark:hidden"
        />
      </>
      <div className="flex flex-row items-center gap-8">
        <div className="flex flex-row items-center gap-3">
          <RxGithubLogo className="w-7 h-7"/>
          <Separator
            className="h-5 w-px bg-border"
            orientation="vertical"
          />
          <PiXLogo className="w-7 h-7"/>
        </div>
        {/* <Button className="bg-card border border-border text-primary"> */}
        <Button variant="ghost" size="sm" className="border border-border">
          <div className="flex flex-row items-center gap-2">
            <MdFeedback className="w-3.5 h-3.5"/>
            Feedback
          </div>
        </Button>
      </div>
    </div>
  )
}
