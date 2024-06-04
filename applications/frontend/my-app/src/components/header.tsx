'use client'
import Image from 'next/image'
import { PiXLogo } from "react-icons/pi";
import { RxGithubLogo } from "react-icons/rx";
import { MdFeedback } from "react-icons/md";
import whiteLogo from '../../public/cc_logo_2white.png'
import blackLogo from '../../public/cc_logo_2.png'

import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { FeedbackDialog } from '@/components/feedback-dialog'

export default function Header() {
  return (
    <div className="flex h-full items-center justify-between text-primary py-2.5 px-4">
      <a href="https://xpaperchallenge.org/cv/" target="_blank">
        <Image
          src={whiteLogo}
          alt="logo"
          sizes="100vw"
          className="w-[40vw] h-auto min-w-36 max-w-80 hidden dark:block"
        />
        <Image
          src={blackLogo}
          alt="logo"
          sizes="100vw"
          className="w-[40vw] h-auto min-w-36 max-w-80 dark:hidden "
        />
      </a>
      <div className="
        flex flex-row items-center
        gap-2
        min-[461px]:max-[600px]:gap-4
        min-[601px]:gap-8
      ">
        <div className="
          flex flex-row items-center
          gap-1
          min-[461px]:gap-3
        ">
          <Button variant="ghost" size="icon" className="w-fit h-fit p-1 rounded-full hover:text-[var(--teal-11)]">
            <a href="https://github.com/cvpaperchallenge" target="_blank">
              <RxGithubLogo className="
                w-4 h-4
                min-[461px]:max-[600px]:w-5 min-[461px]:max-[600px]:h-5
                min-[601px]:w-7 min-[601px]:h-7
              "/>
            </a>
          </Button>
          <Separator
            className="h-5 w-px bg-border"
            orientation="vertical"
          />
          <Button variant="ghost" size="icon" className="w-fit h-fit p-1 aspect-square rounded-full hover:text-[var(--teal-11)]">
            <a href="https://twitter.com/CVpaperChalleng" target="_blank">
              <PiXLogo className="
                w-4 h-4
                min-[461px]:max-[600px]:w-5 min-[461px]:max-[600px]:h-5
                min-[601px]:w-7 min-[601px]:h-7
              "/>
            </a>
          </Button>
        </div>
        <FeedbackDialog />
      </div>
    </div>
  )
}
