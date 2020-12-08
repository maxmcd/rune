//! The state machine assembler of Rune.

use hashbrown::HashMap;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::rc::Rc;
use thiserror::Error;

/// The identifier of a constant.
#[derive(Debug, Clone, Copy)]
pub struct ConstId(usize);

impl fmt::Display for ConstId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "c{}", self.0)
    }
}

/// A variable that can be used as block entries or temporaries.
/// Instructions typically produce and use vars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(usize);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Identifier to a block.
#[derive(Debug, Clone, Copy)]
pub struct BlockId(usize);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "block{}", self.0)
    }
}

/// Error raised during machine construction.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum Error {
    #[error("mismatch in block inputs, expected {expected} but got {actual}")]
    BlockInputMismatch { expected: usize, actual: usize },
}

/// A constant value.
pub enum Constant {
    /// The unit constant (always has constant id = 0).
    Unit,
    /// A boolean constant.
    Bool(bool),
    /// A character constant.
    Char(char),
    /// A byte constant.
    Byte(u8),
    /// An integer constant.
    Integer(i64),
    /// A float constant.
    Float(f64),
    /// A string constant.
    String(Box<str>),
    /// A byte constant.
    Bytes(Box<[u8]>),
}

impl fmt::Debug for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constant::Unit => {
                write!(f, "()")?;
            }
            Constant::Bool(b) => {
                write!(f, "{}", b)?;
            }
            Constant::Char(c) => {
                write!(f, "{:?}", c)?;
            }
            Constant::Byte(b) => {
                write!(f, "0x{:02x}", b)?;
            }
            Constant::Integer(n) => {
                write!(f, "{}", n)?;
            }
            Constant::Float(n) => {
                write!(f, "{}", n)?;
            }
            Constant::String(s) => {
                write!(f, "{:?}", s)?;
            }
            Constant::Bytes(b) => {
                write!(f, "{:?}", b)?;
            }
        }

        Ok(())
    }
}

/// A single abstract machine instruction.
pub enum Inst {
    /// An instruction to load a constant as a value.
    Const(ConstId),
    /// Adds an artificial user of the given value, preventing it from being
    /// pruned. This forces the def-chain to be evaluated for side effects.
    Use(ValueId),
    /// Compute `lhs + rhs`.
    Add(ValueId, ValueId),
    /// Compute `lhs - rhs`.
    Sub(ValueId, ValueId),
    /// Compute `lhs / rhs`.
    Div(ValueId, ValueId),
    /// Compute `lhs * rhs`.
    Mul(ValueId, ValueId),
    /// Conditionally jump to the given block if the given condition is true.
    JumpIf(ValueId, BlockId, Vec<ValueId>),
    /// Return from the current procedure with the given value.
    Return(ValueId),
    /// Compare if `lhs < rhs`.
    CmpLt(ValueId, ValueId),
    /// Compare if `lhs <= rhs`.
    CmpLte(ValueId, ValueId),
    /// Compare if `lhs == rhs`.
    CmpEq(ValueId, ValueId),
    /// Compare if `lhs > rhs`.
    CmpGt(ValueId, ValueId),
    /// Compare if `lhs >= rhs`.
    CmpGte(ValueId, ValueId),
}

impl Inst {
    /// Dump diagnostical information on an instruction.
    pub fn dump(&self) -> InstDump<'_> {
        InstDump(self)
    }
}

pub struct InstDump<'a>(&'a Inst);

impl fmt::Display for InstDump<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Inst::Const(id) => {
                write!(f, "{}", id)?;
            }
            Inst::Use(v) => {
                write!(f, "use {}", v)?;
            }
            Inst::Add(lhs, rhs) => {
                write!(f, "add {}, {}", lhs, rhs)?;
            }
            Inst::Sub(lhs, rhs) => {
                write!(f, "sub {}, {}", lhs, rhs)?;
            }
            Inst::Div(lhs, rhs) => {
                write!(f, "div {}, {}", lhs, rhs)?;
            }
            Inst::Mul(lhs, rhs) => {
                write!(f, "mul {}, {}", lhs, rhs)?;
            }
            Inst::JumpIf(cond, block, vars) => {
                write!(f, "jump-if {}, {}, {}", cond, block, ListDump(vars))?;
            }
            Inst::Return(value) => {
                write!(f, "return {}", value)?;
            }
            Inst::CmpLt(lhs, rhs) => {
                write!(f, "lt {}, {}", lhs, rhs)?;
            }
            Inst::CmpLte(lhs, rhs) => {
                write!(f, "lte {}, {}", lhs, rhs)?;
            }
            Inst::CmpEq(lhs, rhs) => {
                write!(f, "eq {}, {}", lhs, rhs)?;
            }
            Inst::CmpGt(lhs, rhs) => {
                write!(f, "gt {}, {}", lhs, rhs)?;
            }
            Inst::CmpGte(lhs, rhs) => {
                write!(f, "gte {}, {}", lhs, rhs)?;
            }
        }

        Ok(())
    }
}

/// Macro to help build a binary op.
macro_rules! block_binary_op {
    ($name:ident, $variant:ident, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(&self, lhs: ValueId, rhs: ValueId) -> ValueId {
            let value = self.inner.global.value();
            self.inner
                .assignments
                .borrow_mut()
                .insert(value, Inst::$variant(lhs, rhs));
            value
        }
    }
}

/// A block containing a sequence of assignments.
///
/// A block carries a definition of its entry.
/// The entry is the sequence of input variables the block expects.
#[derive(Clone)]
pub struct Block {
    inner: Rc<BlockInner>,
}

impl Block {
    /// Construct a new empty block.
    fn new(global: Global) -> Self {
        let id = global.block();

        Self {
            inner: Rc::new(BlockInner {
                id,
                global,
                inputs: RefCell::new(Vec::new()),
                assignments: RefCell::new(HashMap::new()),
                instructions: RefCell::new(Vec::new()),
                ancestors: RefCell::new(Vec::new()),
            }),
        }
    }

    /// Get the identifier of the block.
    #[inline]
    pub fn id(&self) -> BlockId {
        self.inner.id
    }

    /// Perform a diagnostical dump of a block.
    #[inline]
    pub fn dump(&self) -> BlockDump<'_> {
        BlockDump(self)
    }

    /// Allocate an input variable.
    pub fn input(&self) -> ValueId {
        let value = self.inner.global.value();
        self.inner.inputs.borrow_mut().push(value);
        value
    }

    /// Define a unit.
    pub fn unit(&self) -> ValueId {
        self.constant(Constant::Unit)
    }

    /// Load a constant as a variable.
    pub fn constant(&self, constant: Constant) -> ValueId {
        let value = self.inner.global.value();
        let const_id = self.inner.global.constant(constant);
        self.inner
            .assignments
            .borrow_mut()
            .insert(value, Inst::Const(const_id));
        value
    }

    /// Force a use of the given value.
    pub fn use_(&self, value: ValueId) {
        self.inner.instructions.borrow_mut().push(Inst::Use(value));
    }

    /// Perform a conditional jump to the given block with the specified inputs
    /// if the given condition is true.
    pub fn jump_if(&self, cond: ValueId, block: &Block, input: &[ValueId]) -> Result<(), Error> {
        let expected = block.inner.inputs.borrow().len();

        if expected != input.len() {
            return Err(Error::BlockInputMismatch {
                expected,
                actual: input.len(),
            });
        }

        self.inner
            .instructions
            .borrow_mut()
            .push(Inst::JumpIf(cond, block.id(), input.to_vec()));

        // Mark this block as an ancestor to the block we're jumping to.
        block.inner.ancestors.borrow_mut().push(self.inner.id);
        Ok(())
    }

    block_binary_op!(add, Add, "Compute `lhs + rhs`.");
    block_binary_op!(sub, Sub, "Compute `lhs - rhs`.");
    block_binary_op!(div, Div, "Compute `lhs / rhs`.");
    block_binary_op!(mul, Mul, "Compute `lhs * rhs`.");
    block_binary_op!(cmp_lt, CmpLt, "Compare if `lhs < rhs`.");
    block_binary_op!(cmp_lte, CmpLte, "Compare if `lhs <= rhs`.");
    block_binary_op!(cmp_eq, CmpEq, "Compare if `lhs == rhs`.");
    block_binary_op!(cmp_gt, CmpGt, "Compare if `lhs > rhs`.");
    block_binary_op!(cmp_gte, CmpGte, "Compare if `lhs >= rhs`.");

    /// Unconditionally return from this the procedure this block belongs to.
    pub fn return_unit(&self) {
        let value = self.unit();

        self.inner
            .instructions
            .borrow_mut()
            .push(Inst::Return(value));
    }

    /// Unconditionally return from this the procedure this block belongs to.
    pub fn return_(&self, value: ValueId) {
        self.inner
            .instructions
            .borrow_mut()
            .push(Inst::Return(value));
    }
}

pub struct BlockDump<'a>(&'a Block);

impl fmt::Display for BlockDump<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inputs = self.0.inner.inputs.borrow();
        let ancestors = self.0.inner.ancestors.borrow();

        if inputs.len() == 0 {
            write!(f, "{}", self.0.id())?;
        } else {
            write!(f, "{}(", self.0.id())?;

            let mut it = inputs.iter();
            let last = it.next_back();

            for v in it {
                write!(f, "{}, ", v)?;
            }

            if let Some(v) = last {
                write!(f, "{}", v)?;
            }

            write!(f, ")")?;
        }

        if ancestors.is_empty() {
            writeln!(f, ":")?;
        } else {
            writeln!(f, " <- {}:", ListDump(&ancestors[..]))?;
        }

        for (v, inst) in self.0.inner.assignments.borrow().iter() {
            writeln!(f, "  {} <- {}", v, inst.dump())?;
        }

        for inst in self.0.inner.instructions.borrow().iter() {
            writeln!(f, "  {}", inst.dump())?;
        }

        Ok(())
    }
}

struct BlockInner {
    /// The identifier of the block.
    id: BlockId,
    /// Global shared stack machine state.
    global: Global,
    /// Input variables.
    inputs: RefCell<Vec<ValueId>>,
    /// Instructions being built.
    assignments: RefCell<HashMap<ValueId, Inst>>,
    /// Instructions that do not produce a value.
    instructions: RefCell<Vec<Inst>>,
    /// Ancestor blocks.
    ancestors: RefCell<Vec<BlockId>>,
}

/// Global construction state of the state machine.
#[derive(Debug, Clone, Default)]
struct Global {
    inner: Rc<GlobalInner>,
}

impl Global {
    /// Allocate a global variable.
    fn value(&self) -> ValueId {
        let id = self.inner.value.get();
        self.inner.value.set(id + 1);
        ValueId(id)
    }

    /// Allocate a global block identifier.
    fn block(&self) -> BlockId {
        let id = self.inner.block.get();
        self.inner.block.set(id + 1);
        BlockId(id)
    }

    /// Allocate a constant.
    fn constant(&self, constant: Constant) -> ConstId {
        let mut constants = self.inner.constants.borrow_mut();

        match &constant {
            Constant::Unit => return ConstId(0),
            Constant::String(s) => {
                let mut string_rev = self.inner.constant_string_rev.borrow_mut();

                if let Some(const_id) = string_rev.get(s) {
                    return *const_id;
                }

                let const_id = ConstId(constants.len());
                string_rev.insert(s.clone(), const_id);
                constants.push(constant);
                return const_id;
            }
            Constant::Bytes(b) => {
                let mut bytes_rev = self.inner.constant_bytes_rev.borrow_mut();

                if let Some(const_id) = bytes_rev.get(b) {
                    return *const_id;
                }

                let const_id = ConstId(constants.len());
                bytes_rev.insert(b.clone(), const_id);
                constants.push(constant);
                return const_id;
            }
            _ => (),
        }

        let const_id = ConstId(constants.len());
        constants.push(constant);
        const_id
    }
}

/// Inner state of the global.
#[derive(Debug)]
struct GlobalInner {
    /// Variable allocator.
    value: Cell<usize>,
    /// Block allocator.
    block: Cell<usize>,
    /// The values of constants.
    constants: RefCell<Vec<Constant>>,
    /// Constant strings that have already been allocated.
    constant_string_rev: RefCell<HashMap<Box<str>, ConstId>>,
    /// Constant byte arrays that have already been allocated.
    constant_bytes_rev: RefCell<HashMap<Box<[u8]>, ConstId>>,
}

impl Default for GlobalInner {
    fn default() -> Self {
        Self {
            value: Default::default(),
            block: Default::default(),
            constants: RefCell::new(vec![Constant::Unit]),
            constant_string_rev: Default::default(),
            constant_bytes_rev: Default::default(),
        }
    }
}

/// The central state machine assembler.
pub struct Program {
    global: Global,
    blocks: Vec<Block>,
}

impl Program {
    /// Construct a new empty state machine.
    pub fn new() -> Self {
        Self {
            global: Global::default(),
            blocks: Vec::new(),
        }
    }

    /// Dump the current state of the program.
    ///
    /// This is useful for diagnostics.
    pub fn dump(&self) -> ProgramDump<'_> {
        ProgramDump(self)
    }

    /// Construct a new block associated with the state machine.
    pub fn block(&mut self) -> Block {
        let block = Block::new(self.global.clone());
        self.blocks.push(block.clone());
        block
    }
}

pub struct ProgramDump<'a>(&'a Program);

impl fmt::Display for ProgramDump<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let constants = self.0.global.inner.constants.borrow();

        if !constants.is_empty() {
            writeln!(f, "constants:")?;

            for (id, c) in constants.iter().enumerate() {
                writeln!(f, "  c{} <- {:?}", id, c)?;
            }

            if !self.0.blocks.is_empty() {
                writeln!(f)?;
            }
        }

        let mut it = self.0.blocks.iter();
        let last = it.next_back();

        for b in it {
            writeln!(f, "{}", b.dump())?;
        }

        if let Some(b) = last {
            write!(f, "{}", b.dump())?;
        }

        Ok(())
    }
}

/// Helper to format the given args as a list.
struct ListDump<I>(I);

impl<I> fmt::Display for ListDump<I>
where
    I: Copy + IntoIterator,
    I::Item: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut it = self.0.into_iter().peekable();

        if it.peek().is_none() {
            write!(f, "[]")?;
            return Ok(());
        }

        write!(f, "[")?;

        while let Some(item) = it.next() {
            write!(f, "{}", item)?;

            if it.peek().is_some() {
                write!(f, ", ")?;
            }
        }

        write!(f, "]")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Constant, Error, Program};

    #[test]
    fn test_basic_sm() -> Result<(), Error> {
        let mut sm = Program::new();

        let mut block = sm.block();
        let mut then = sm.block();

        let else_value = block.constant(Constant::Integer(1));
        then.return_(else_value);

        // Define one input variable to the block.
        let a = block.input();
        let b = block.constant(Constant::Integer(42));
        let unit = block.constant(Constant::Unit);
        let c = block.add(a, b);

        let d = block.cmp_lt(c, b);
        block.jump_if(d, &then, &[])?;
        block.return_(unit);

        println!("{}", sm.dump());
        Ok(())
    }
}
