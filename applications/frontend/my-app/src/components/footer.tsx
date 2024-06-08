'use client'
import Image from 'next/image'
import { Separator } from "@/components/ui/separator"
import { RxGithubLogo } from "react-icons/rx";
import whiteLogo from '../../public/cc_logo_2white.png'
import blackLogo from '../../public/cc_logo_2.png'
import forwardPropagationMark from '../../public/forward-propergation-chan4.png'

export default function Footer() {
  const year = new Date().getFullYear()
  return (
    <div className="
      flex
      h-full
      px-4 min-[461px]:max-[600px]:px-8 min-[601px]:px-8
      py-7 min-[461px]:max-[600px]:px-10 min-[601px]:px-12
      items-center
      justify-between
      text-[var(--olive-11)]
    ">
      <div className="
        flex
        flex-col min-[861px]:flex-row
        items-start min-[861px]:items-center
        gap-3
      ">
        <div className="flex flex-col items-start gap-2">
          <a href="https://xpaperchallenge.org/cv/" target="_blank">
            <Image
              src={whiteLogo}
              alt="logo"
              sizes="100vw"
              className="w-[40vw] h-auto min-w-36 max-w-56 hidden dark:block"
            />
            <Image
              src={blackLogo}
              alt="logo"
              sizes="100vw"
              className="w-[40vw] h-auto min-w-36 max-w-80 dark:hidden "
            />
          </a>
          <div className='pl-2 text-xs min-[461px]:text-sm'>
            &copy; 2015-{year}
          </div>
        </div>
        <Separator
          className="
            ml-2
            h-px min-[861px]:h-5
            w-5 min-[861px]:w-px
            bg-border
          "
          orientation="horizontal"
        />
        <div className="pl-2 text-xs min-[461px]:text-sm">
          Supported by <span className="inline">
            <a href="https://confit.atlas.jp/guide/event/ssii2024/top?lang=ja" className="visited:text-purple-600">
              SSII 2024
            </a>
          </span>
        </div>
      </div>
      <div className="
        flex
        flex-row
        items-center
        rounded-md
        min-w-[180px]
        bg-[var(--black-a2)] dark:bg-[var(--white-a1)]
        px-2.5 min-[611px]:px-7
        py-2.5
        gap-0 min-[611px]:gap-4
      ">
        <div className="flex flex-col items-start gap-2">
          <div className="font-semibold text-sm min-[611px]:text-base">
            Developed by
          </div>
          <div className="flex flex-col items-start min-w-[140px] gap-1">
            <a href="https://github.com/YoshikiKubotani">
              <div className="flex flex-row items-center">
                <RxGithubLogo className="w-4 h-4 text-primary"/>
                <span className="pl-1 text-xs min-[611px]:text-sm">
                  Yoshiki Kubotani
                </span>
              </div>
            </a>
            <a href="https://github.com/gatheluck">
              <div className="flex flex-row items-center">
                <RxGithubLogo className="w-4 h-4 text-primary"/>
                <span className="pl-1 text-xs min-[611px]:text-sm">
                  Yoshihiro Fukuhara
                </span>
              </div>
            </a>
            <a href="https://github.com/Hina39">
              <div className="flex flex-row items-center">
                <RxGithubLogo className="w-4 h-4 text-primary"/>
                <span className="pl-1 text-xs min-[611px]:text-sm">
                  Hina Otake
                </span>
              </div>
            </a>
          </div>
        </div>
        <Image
          src={forwardPropagationMark}
          alt="jundenpa_chan"
          sizes="100vw"
          className='
            w-5 min-[461px]:max-[610px]:w-7 min-[611px]:w-10
            h-5 min-[461px]:max-[610px]:h-7 min-[611px]:h-10
          '/>
      </div>
    </div>
  )
}
