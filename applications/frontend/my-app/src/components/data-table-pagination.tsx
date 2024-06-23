import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination"

import { Table } from "@tanstack/react-table"
import { RxCaretLeft, RxCaretRight, RxDoubleArrowLeft, RxDoubleArrowRight } from "react-icons/rx";

interface DataTablePaginationProps<TData> {
  table: Table<TData>
  numPagesDisplayed: number
}

export function DataTablePagination<TData>({
  table,
  numPagesDisplayed
}: DataTablePaginationProps<TData>) {
  const getPaginationItems = () => {
    const pageCount = table.getPageCount();
    const pageIndex = table.getState().pagination.pageIndex + 1;
    const items = [];
    let startPage = Math.max(1, pageIndex - Math.floor(numPagesDisplayed / 2));
    let endPage = Math.min(pageCount, startPage + numPagesDisplayed - 1);

    // Adjust start and end if we're too close to the boundaries
    if (startPage === 1) {
      endPage = Math.min(pageCount, startPage + numPagesDisplayed - 1);
    } else if (endPage === pageCount) {
      startPage = Math.max(1, endPage - numPagesDisplayed + 1);
    }
    // Add ellipsis if necessary at the beginning
    if (startPage > 1) {
      items.push(
          <PaginationItem key="start-ellipsis">
              <PaginationEllipsis />
          </PaginationItem>
      );
    }

    for (let i = startPage; i <= endPage; i++) {
        items.push(
            <PaginationItem>
                <PaginationLink
                    isActive={i === pageIndex}
                    onClick={() => {
                        table.setPageIndex(i - 1);
                    }}
                >
                    {i}
                </PaginationLink>
            </PaginationItem>
        );
    }

    // Add ellipsis if necessary at the end
    if (endPage < pageCount) {
      items.push(
          <PaginationItem key="end-ellipsis">
              <PaginationEllipsis />
          </PaginationItem>
      );
    }
    return items;
  }

  return (
    <Pagination className="p-3 bg-[var(--teal-4)] dark:bg-[var(--teal-3)] border-t">
      <PaginationContent>
        <PaginationItem>
          <PaginationLink
            aria-label="Go to the first page"
            size="default"
            onClick={() => table.firstPage()}
            className="gap-1 pl-2.5"
            isDisabled={!table.getCanPreviousPage()}
          >
            <RxDoubleArrowLeft className="h-4 w-4" />
          </PaginationLink>
          <PaginationLink
            aria-label="Go to previous page"
            size="default"
            onClick={() => table.previousPage()}
            className="gap-1 pl-2.5"
            isDisabled={!table.getCanPreviousPage()}
          >
            <RxCaretLeft className="h-4 w-4" />
          </PaginationLink>
        </PaginationItem>
        {getPaginationItems()}
        <PaginationItem>
          <PaginationLink
            aria-label="Go to next page"
            size="default"
            onClick={() => table.nextPage()}
            className="gap-1 pr-2.5"
            isDisabled={!table.getCanNextPage()}
          >
            <RxCaretRight className="h-4 w-4" />
          </PaginationLink>
          <PaginationLink
            aria-label="Go to the last page"
            size="default"
            onClick={() => table.lastPage()}
            className="gap-1 pr-2.5"
            isDisabled={!table.getCanNextPage()}
          >
            <RxDoubleArrowRight className="h-4 w-4" />
          </PaginationLink>
        </PaginationItem>
      </PaginationContent>
    </Pagination>
  )
}