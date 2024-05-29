'use client'
import Image from 'next/image'
import { PaperPlaneIcon, ChatBubbleIcon } from '@radix-ui/react-icons'
import {Dialog, Flex, Button, Text, TextField, Box, TextArea} from '@radix-ui/themes'

export default function Header() {
  const handleSendFeedback = () => {
    alert('Feedback sent!')
  }
  return (
    <section
      style={{
        background: 'var(--accent-2)',
      }}>
      <Flex
        direction="row"
        justify="between"
        align="center"
        style={{
          flexShrink: 0,
          padding: "10px 15px",
          margin: "0 auto",
          width: "1100px",
          maxWidth: "100%",
        }}
      >
        <Image
          src="/cc_logo_2white.png"
          alt="logo"
          width={300}
          height={35}
          style={{
            objectFit: 'cover',
          }}
        />
        <Box py="20px">
          <Dialog.Root>
            <Dialog.Trigger>
              <Button variant="surface">
                <PaperPlaneIcon/>
                Send feedback
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
                  />
                </label>
                <label>
                  <Text as="div" size="2" mb="1" weight="bold">
                    フィードバック <Text color="gray" size="1">/ Feedback</Text>
                  </Text>
                  <TextArea
                    placeholder="Drop your feedback here!"
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
                  <Button onClick={handleSendFeedback} color="mint">Send</Button>
                </Dialog.Close>
              </Flex>
            </Dialog.Content>
          </Dialog.Root>
        </Box>
      </Flex>
    </section>
  )
}