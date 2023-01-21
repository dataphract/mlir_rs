use std::marker::PhantomData;

use crate::{ffi, Block, BlockRef, RegionRef};

// NOTE: Deliberately not Send/Sync.
pub struct BlockCursor<'region> {
    region: RegionRef<'region>,
    block: ffi::MlirBlock,
}

impl<'region> BlockCursor<'region> {
    /// Returns a reference to the block the cursor is pointing to.
    ///
    /// If the cursor is pointing to the null element, returns `None`.
    pub fn get(&self) -> Option<BlockRef> {
        unsafe { BlockRef::from_raw(self.block) }
    }

    /// Detaches and returns the block the cursor is pointing to.
    ///
    /// If the cursor is pointing to the null element, returns `None`.
    pub fn detach(&mut self) -> Option<Block> {
        let to_detach = self.block;
        if to_detach.ptr.is_null() {
            return None;
        }

        self.move_next();
        unsafe {
            ffi::mlirBlockDetach(to_detach);
            Some(Block::from_raw(to_detach).unwrap())
        }
    }

    /// Inserts a block before the block the cursor is pointing to.
    ///
    /// If the cursor is pointing to the null element, appends the block to the region.
    pub fn insert_before(&mut self, block: Block) {
        unsafe {
            ffi::mlirRegionInsertOwnedBlockBefore(self.region.as_raw(), self.block, block.as_raw())
        };
    }

    /// Inserts a block after the block the cursor is pointing to.
    ///
    /// If the cursor is pointing to the null element, prepends the block to the region.
    pub fn insert_after(&mut self, block: Block) {
        unsafe {
            ffi::mlirRegionInsertOwnedBlockAfter(self.region.as_raw(), self.block, block.as_raw())
        };
    }

    /// Points the cursor to the next block in the region.
    ///
    /// If the cursor was pointing to the null element, calling this method points it to the first
    /// block in the region.
    pub fn move_next(&mut self) {
        self.block = unsafe { ffi::mlirBlockGetNextInRegion(self.block) };
    }

    // TODO: not exposed by the C API.
    // pub fn move_prev(&mut self) {
    //     self.block = unsafe { ffi::mlirBlockGetPrevInRegion(self.block) };
    // }
}
