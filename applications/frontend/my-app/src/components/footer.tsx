'use client'
import Image from 'next/image'
import { RxGithubLogo } from 'react-icons/rx'

import { Separator } from '@/components/ui/separator'

import blackLogo from '../../public/cc_logo_2.png'
import whiteLogo from '../../public/cc_logo_2white.png'
import whiteForwardPropagationMark from '../../public/forward-propergation-chan4-white.png'
import blackForwardPropagationMark from '../../public/forward-propergation-chan4.png'

export default function Footer() {
  const year = new Date().getFullYear()
  return (
    <div
      className="
      flex
      h-full
      items-center justify-between px-4
      py-7 text-[var(--olive-11)] min-[461px]:max-[600px]:px-8
      min-[461px]:max-[600px]:py-10
      min-[601px]:px-8
      min-[601px]:py-12
    "
    >
      <div
        className="
        flex
        flex-col items-start
        gap-3 min-[861px]:flex-row
        min-[861px]:items-center
      "
      >
        <div className="flex flex-col items-start gap-2">
          <a href="https://xpaperchallenge.org/cv/" target="_blank">
            <Image
              alt="logo"
              priority={true}
              className="hidden h-auto w-[35vw] min-w-36 max-w-56 dark:block"
              sizes="100vw"
              src={whiteLogo}
            />
            <Image
              alt="logo"
              priority={true}
              className="h-auto w-[35vw] min-w-36 max-w-56 dark:hidden "
              sizes="100vw"
              src={blackLogo}
            />
          </a>
          <div className="pl-2 text-xs min-[461px]:text-sm">
            &copy; 2015-{year}
          </div>
        </div>
        <Separator
          className="
            ml-2
            h-px w-5
            bg-[var(--olive-8)] min-[861px]:h-5
            min-[861px]:w-px
          "
          orientation="horizontal"
        />
        <div className="pl-2 text-xs min-[461px]:text-sm">
          Supported by{' '}
          <span className="inline">
            <a
              className="text-primary hover:border-b-2 hover:bg-accent hover:text-accent-foreground active:border-b-2 active:bg-primary active:text-primary-foreground transition-colors"
              href="https://confit.atlas.jp/guide/event/ssii2024/top?lang=ja"
              target="_blank"
            >
              SSII 2024
            </a>
          </span>
        </div>
      </div>
      <div
        className="
        flex
        min-w-[170px]
        flex-row
        items-center
        gap-0
        rounded-md bg-[var(--black-a2)]
        p-2.5 dark:bg-[var(--white-a1)]
        min-[611px]:gap-4
        min-[611px]:px-7
      "
      >
        <div className="flex flex-col items-start gap-2">
          <div className="text-sm font-semibold text-card-foreground min-[611px]:text-base">
            Developed by
          </div>
          <div className="flex min-w-[130px] flex-col items-start gap-1">
            <a
              className="p-0.5 hover:bg-[var(--teal-4)] hover:text-foreground active:bg-primary active:text-primary-foreground transition-colors"
              href="https://github.com/YoshikiKubotani"
              target="_blank"
            >
              <div className="flex flex-row items-center">
                <RxGithubLogo className="mr-1 size-4" />
                <span className="text-[10px] min-[611px]:text-sm">
                  Yoshiki Kubotani
                </span>
              </div>
            </a>
            <a
              className="p-0.5 hover:bg-[var(--teal-4)] hover:text-foreground active:bg-primary active:text-primary-foreground transition-colors"
              href="https://github.com/gatheluck"
              target="_blank"
            >
              <div className="flex flex-row items-center">
                <RxGithubLogo className="mr-1 size-4" />
                <span className="text-[10px] min-[611px]:text-sm">
                  Yoshihiro Fukuhara
                </span>
              </div>
            </a>
            <a
              className="p-0.5 hover:bg-[var(--teal-4)] hover:text-foreground active:bg-primary active:text-primary-foreground transition-colors"
              href="https://github.com/Hina39"
              target="_blank"
            >
              <div className="flex flex-row items-center">
                <RxGithubLogo className="mr-1 size-4" />
                <span className="text-[10px] min-[611px]:text-sm">
                  Hina Otake
                </span>
              </div>
            </a>
          </div>
        </div>
        <Image
          alt="jundenpa_chan"
          priority={true}
          className="
            hidden size-5 dark:block
            min-[461px]:max-[610px]:size-7 min-[611px]:size-10
          "
          sizes="100vw"
          src={whiteForwardPropagationMark}
        />
        <Image
          alt="jundenpa_chan"
          priority={true}
          className="
            size-5 dark:hidden min-[461px]:max-[610px]:size-7
            min-[611px]:size-10
          "
          sizes="100vw"
          src={blackForwardPropagationMark}
        />
      </div>
    </div>
  )
}
