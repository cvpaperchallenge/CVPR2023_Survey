import { MdFeedback } from "react-icons/md";
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogClose
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"

export function FeedbackDialog() {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" className="border border-border text-xs min-[461px]:text-sm hover:text-[var(--teal-11)]">
          <MdFeedback className="
            mr-2
            w-3 h-3
            min-[601px]:w-4 min-[601px]:h-4
          "/>
          Feedback
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px] bg-[var(--teal-3)] dark:bg-[var(--teal-2)]">
        <DialogHeader>
          <DialogTitle className="text-left">
            <MdFeedback className="mr-2 w-auto h-5 inline"/> Please send your feedback!
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
            <Label htmlFor="name" className="text-left">
              <span className="inline text-[var(--olive-12)]">
                名前
              </span>
              <span className="pl-1 inline text-xs text-[var(--olive-11)]">
                / Name
              </span>
            </Label>
            <Input
              id="name"
              placeholder="Enter your name"
            />
          </div>
          <div className="flex flex-col gap-1">
            <Label htmlFor="feedback" className="text-left">
              <span className="inline text-[var(--olive-12)]">
                フィードバック
              </span>
              <span className="pl-1 inline text-xs text-[var(--olive-11)]">
                / Feedback
              </span>
            </Label>
            <Textarea
              id="feedback"
              className="resize-none"
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
          <Button type="submit" variant="default" className="hover:bg-[var(--teal-a12)]">Send</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
