import {Tabs, Box} from '@radix-ui/themes'

import { Paper } from '../libs/types';
import PaperList from './paper-list';

interface PaperListBoardProps {
  papers: Paper[];
  conferenceName: string;
  setConferenceName: (name: string) => void;
}

export default function PaperListBoard({
  papers,
  conferenceName,
  setConferenceName,
}: PaperListBoardProps) {
  const handleTabChange = (value: string) => {
    setConferenceName(value);
  };

  return (
    <Tabs.Root defaultValue="cvpr-2023" onValueChange={handleTabChange}>
      <Tabs.List>
        <Tabs.Trigger value="cvpr-2023">CVPR2023</Tabs.Trigger>
        <Tabs.Trigger value="iccv-2023">ICCV2023</Tabs.Trigger>
      </Tabs.List>

      <Box pt="3">
        <Tabs.Content value="cvpr-2023">
          <PaperList papers={papers} conferenceName={conferenceName} />
        </Tabs.Content>

        <Tabs.Content value="iccv-2023">
          <PaperList papers={papers} conferenceName={conferenceName} />
        </Tabs.Content>
      </Box>
    </Tabs.Root>
  );
};
