'use client'
import Image from 'next/image'
import { useTheme } from 'next-themes'
import { PiXLogo } from 'react-icons/pi'
import { RxGithubLogo, RxMoon, RxSun } from 'react-icons/rx'

import { FeedbackDialog } from '@/components/feedback-dialog'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Switch } from '@/components/ui/switch'

import blackLogo from '../../public/cc_logo_2.png'
import whiteLogo from '../../public/cc_logo_2white.png'

export default function Header() {
  const { theme, setTheme } = useTheme()
  return (
    <div className="flex h-full items-center justify-between px-4 py-2.5 text-primary">
      <a href="https://xpaperchallenge.org/cv/" target="_blank">
        <Image
          alt="logo"
          priority={true}
          className="hidden h-auto w-[35vw] min-w-36 max-w-80 dark:block"
          sizes="100vw"
          src={whiteLogo}
        />
        <Image
          alt="logo"
          priority={true}
          className="h-auto w-[35vw] min-w-36 max-w-80 dark:hidden "
          sizes="100vw"
          src={blackLogo}
        />
      </a>
      <div
        className="
        flex flex-row items-center
        gap-2
        min-[461px]:max-[600px]:gap-4
        min-[601px]:gap-6
      ">
        <div className="
          flex
          flex-col-reverse min-[501px]:flex-row
          items-center
          gap-2 min-[501px]:max-[600px]:gap-4 min-[601px]:gap-6
        ">
          <div className="flex items-center space-x-1 min-[461px]:space-x-2 max-[501px]:hidden">
            <RxSun className="
              size-3
              min-[461px]:max-[600px]:size-3.5
              min-[601px]:size-4
            "/>
            <Switch
              id="airplane-mode"
              checked={theme === 'dark'}
              onCheckedChange={(checked) => {
                setTheme(checked ? 'dark' : 'light')
              }}
              className="
              w-8 min-[461px]:max-[600px]:w-10 min-[601px]:w-11
              h-4 min-[461px]:max-[600px]:h-5 min-[601px]:h-6
            "/>
            <RxMoon className="
              size-3
              min-[461px]:max-[600px]:size-3.5
              min-[601px]:size-4
            "/>
          </div>
          <div
            className="
            flex flex-row items-center
            gap-1
            min-[461px]:max-[600px]:gap-1
            min-[601px]:gap-2
          "
          >
            <Button
              className="size-fit rounded-full p-1 hover:text-[var(--teal-11)]"
              size="icon"
              variant="ghost"
            >
              <a href="https://github.com/cvpaperchallenge" target="_blank">
                <RxGithubLogo
                  className="
                  size-4 min-[461px]:max-[600px]:size-5
                  min-[601px]:size-7
                "/>
              </a>
            </Button>
            <Separator
              className="h-5 w-px bg-[var(--olive-8)]"
              orientation="vertical"
            />
            <Button
              className="aspect-square size-fit rounded-full p-1 hover:text-[var(--teal-11)]"
              size="icon"
              variant="ghost"
            >
              <a href="https://twitter.com/CVpaperChalleng" target="_blank">
                <PiXLogo
                  className="
                  size-4 min-[461px]:max-[600px]:size-5
                  min-[601px]:size-7
                "
                />
              </a>
            </Button>
          </div>
        </div>
        <FeedbackDialog />
      </div>
    </div>
  )
}
