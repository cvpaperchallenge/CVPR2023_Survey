import { MagnifyingGlassIcon, DotsHorizontalIcon } from '@radix-ui/react-icons'
import {Box, Flex, Text, TextField, Tabs} from '@radix-ui/themes'

export default function PaperListLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <section>
      <Flex direction="column" justify="center" align="center" gap="3">
        {children}
      </Flex>
    </section>
  )
}