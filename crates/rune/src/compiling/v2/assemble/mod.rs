use crate::compiling::v2::Compiler;
use crate::compiling::CompileResult;
use rune_im::{Block, ValueId};

mod block;
mod expr;
mod item_fn;
mod prelude;

/// Compiler trait implemented for things that can be compiled.
///
/// This is the new compiler trait to implement.
pub(crate) trait Assemble {
    /// Walk the current type with the given item.
    fn assemble(&self, block: &Block, c: &mut Compiler<'_>) -> CompileResult<ValueId>;
}

/// Assemble a function.
pub(crate) trait AssembleFn {
    /// Assemble a function.
    fn assemble_fn(&self, c: &mut Compiler<'_>, instance_fn: bool) -> CompileResult<Block>;
}
