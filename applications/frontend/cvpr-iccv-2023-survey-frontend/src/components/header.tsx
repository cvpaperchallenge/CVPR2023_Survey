'use client'
import Image from 'next/image'
import { PaperPlaneIcon, GitHubLogoIcon, TwitterLogoIcon, ChatBubbleIcon } from '@radix-ui/react-icons'
import {Dialog, Flex, Button, Text, TextField, Box, TextArea, IconButton, Link, Separator} from '@radix-ui/themes'

import "./header-styles.css"
import { useState } from 'react'
import { sendFeedback, handleFetchResult} from '@/libs/fetch'

export default function Header() {
  const [name, setName] = useState('')
  const [feedback, setFeedback] = useState('')

  const handleSendFeedback = (async () => {
    const result = await sendFeedback(name, feedback)
    handleFetchResult<null>(result, 'Failed to send feedback', 'Feedback sent successfully!')
    setName('')
    setFeedback('')
  })

  const isError = /^\s*$/.test(name) || /^\s*$/.test(feedback)

  return (
    <section
      style={{
        background: 'var(--accent-2)',
        minWidth: '350px',
      }}>
      <Flex
        direction="row"
        justify="between"
        align="center"
        minWidth="350px"
        maxWidth="100%"
        width="100vw"
        mx="auto"
        px="15px"
        py="10px"
        className='header'
        style={{
          flexShrink: 0,
      }}
      >
        <Link href="https://xpaperchallenge.org/cv/">
          <Image
            src="/cc_logo_2white.png"
            alt="logo"
            width={300}
            height={35}
            className='logo'
            style={{
              objectFit: 'cover',
            }}
          />
        </Link>
        <Flex direction="row" align="center" gap="6" className='icon-feedback-gap'>
          <Flex direction="row" align="center" gap="3" className="icons-gap">
            <IconButton variant="ghost" asChild>
              <Link href="https://github.com/cvpaperchallenge">
                <GitHubLogoIcon className="icon-size" />
              </Link>
            </IconButton>
            <Separator orientation="vertical" size="1"/>
            <IconButton variant="ghost" asChild>
              <Link href="https://twitter.com/CVpaperChalleng">
                <TwitterLogoIcon className="icon-size" />
              </Link>
            </IconButton>
          </Flex>
          <Box py="20px">
            <Dialog.Root>
              <Dialog.Trigger>
                <Button variant="surface" size="3" className='feedback-button-size'>
                  <PaperPlaneIcon className='feedback-icon-size'/>
                  Feedback
                </Button>
              </Dialog.Trigger>

              <Dialog.Content maxWidth="450px" style={{backgroundColor: "var(--mint-1)"}}>
                <Dialog.Title color="mint">
                  <Flex direction="row" align="center" gap="3">
                    <ChatBubbleIcon style={{ width: '20px', height: '20px' }}/>
                    Please send your feedback!
                  </Flex>
                </Dialog.Title>
                <Dialog.Description size="2" mb="4">
                  機能要望や使ってみての感想など、フィードバックがあればご記入ください。<br/>
                  <Text color="gray" size="1">Fill in your feedback, such as feature requests or impressions.</Text>
                </Dialog.Description>

                <Flex direction="column" gap="3">
                  <label>
                    <Text as="div" size="2" mb="1" weight="bold">
                      名前 <Text color="gray" size="1">/ Name</Text>
                    </Text>
                    <TextField.Root
                      placeholder="Enter your name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                    />
                  </label>
                  <label>
                    <Text as="div" size="2" mb="1" weight="bold">
                      フィードバック <Text color="gray" size="1">/ Feedback</Text>
                    </Text>
                    <TextArea
                      placeholder="Drop your feedback here!"
                      value={feedback}
                      onChange={(e) => setFeedback(e.target.value)}
                    />
                  </label>
                </Flex>

                <Flex gap="3" mt="4" justify="end">
                  <Dialog.Close>
                    <Button variant="soft" color="gray">
                      Cancel
                    </Button>
                  </Dialog.Close>
                  <Dialog.Close>
                    <Button onClick={handleSendFeedback} color="mint" disabled={isError}>Send</Button>
                  </Dialog.Close>
                </Flex>
              </Dialog.Content>
            </Dialog.Root>
          </Box>
        </Flex>
      </Flex>
    </section>
  )
}