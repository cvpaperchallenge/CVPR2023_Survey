import { MdFeedback } from 'react-icons/md'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogClose,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'

export function FeedbackDialog() {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          className="border border-border text-xs hover:text-[var(--teal-11)] min-[461px]:text-sm"
          size="sm"
          variant="ghost"
        >
          <MdFeedback
            className="
            mr-2
            size-3 min-[601px]:size-4
          "
          />
          Feedback
        </Button>
      </DialogTrigger>
      <DialogContent className="bg-[var(--teal-3)] dark:bg-[var(--teal-2)] sm:max-w-[450px]">
        <DialogHeader>
          <DialogTitle className="text-left">
            <MdFeedback className="mr-2 inline h-5 w-auto" /> Please send your
            feedback!
          </DialogTitle>
          <DialogDescription className="text-left">
            <div className="text-[var(--olive-12)]">
              機能要望や使ってみての感想など、フィードバックがあればご記入ください。
            </div>
            <div className="text-xs text-[var(--olive-11)]">
              Fill in your feedback, such as feature requests or impressions.
            </div>
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-4 py-1">
          <div className="flex flex-col gap-1">
            <Label className="text-left" htmlFor="name">
              <span className="inline text-[var(--olive-12)]">名前</span>
              <span className="inline pl-1 text-xs text-[var(--olive-11)]">
                / Name
              </span>
            </Label>
            <Input id="name" placeholder="Enter your name" />
          </div>
          <div className="flex flex-col gap-1">
            <Label className="text-left" htmlFor="feedback">
              <span className="inline text-[var(--olive-12)]">
                フィードバック
              </span>
              <span className="inline pl-1 text-xs text-[var(--olive-11)]">
                / Feedback
              </span>
            </Label>
            <Textarea
              className="resize-none"
              id="feedback"
              placeholder="Drop your feedback here!"
            />
          </div>
        </div>
        <DialogFooter className="flex flex-row justify-end gap-2">
          <DialogClose>
            <Button type="button" variant="outline">
              Cancel
            </Button>
          </DialogClose>
          <Button
            className="hover:bg-[var(--teal-a12)]"
            type="submit"
            variant="default"
          >
            Send
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
