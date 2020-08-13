use crate::value::Slot;
use crate::vm::VmError;
use std::cell::Cell;
use std::fmt;
use std::marker;
use std::ops;

#[derive(Debug, Clone)]
pub(super) struct Access(Cell<isize>);

impl Access {
    /// Construct a new default access.
    pub const fn new() -> Self {
        Self(Cell::new(0))
    }

    /// Test if we have shared access without modifying the internal count.
    #[inline]
    pub(super) fn test_shared(&self, slot: Slot) -> Result<(), VmError> {
        let b = self.0.get().wrapping_sub(1);

        if b >= 0 {
            return Err(VmError::SlotInaccessibleShared { slot });
        }

        Ok(())
    }

    /// Mark that we want shared access to the given access token.
    #[inline]
    pub(super) fn shared(&self, slot: Slot) -> Result<(), VmError> {
        let b = self.0.get().wrapping_sub(1);

        if b >= 0 {
            return Err(VmError::SlotInaccessibleShared { slot });
        }

        self.0.set(b);
        Ok(())
    }

    /// Unshare the current access.
    #[inline]
    pub(super) fn release_shared(&self) {
        let b = self.0.get().wrapping_add(1);
        debug_assert!(b <= 0);
        self.0.set(b);
    }

    /// Unshare the current access.
    #[inline]
    pub(super) fn release_exclusive(&self) {
        let b = self.0.get().wrapping_sub(1);
        debug_assert!(b == 0);
        self.0.set(b);
    }

    /// Mark that we want exclusive access to the given access token.
    #[inline]
    pub(super) fn exclusive(&self, slot: Slot) -> Result<(), VmError> {
        let b = self.0.get().wrapping_add(1);

        if b != 1 {
            return Err(VmError::SlotInaccessibleExclusive { slot });
        }

        self.0.set(b);
        Ok(())
    }
}

/// A raw reference guard.
pub struct RawRefGuard {
    pub(super) access: *const Access,
}

impl Drop for RawRefGuard {
    fn drop(&mut self) {
        unsafe { (*self.access).release_shared() };
    }
}

/// A raw guard for borrowed values.
pub(super) struct RawRef<T: ?Sized> {
    pub(super) value: *const T,
    pub(super) guard: RawRefGuard,
}

/// Guard for a value borrowed from a slot in the virtual machine.
///
/// These guards are necessary, since we need to guarantee certain forms of
/// access depending on what we do. Releasing the guard releases the access.
///
/// These also aid in function call integration, since they can be "arm" the
/// virtual machine to release shared guards through its unsafe functions.
///
/// See [clear] for more information.
///
/// [clear]: [crate::Vm::clear]
pub struct Ref<'a, T: ?Sized + 'a> {
    pub(super) raw: RawRef<T>,
    pub(super) _marker: marker::PhantomData<&'a T>,
}

impl<'a, T: ?Sized> Ref<'a, T> {
    /// Convert into a raw pointer and associated raw access guard.
    ///
    /// # Safety
    ///
    /// The returned pointer must not outlive the associated guard, since this
    /// prevents other uses of the underlying data which is incompatible with
    /// the current.
    ///
    /// The returned pointer also must not outlive the VM that produced.
    /// Nor a call to clear the VM using [clear], since this will free up the
    /// data being referenced.
    ///
    /// [clear]: [crate::Vm::clear]
    pub unsafe fn unsafe_into_ref(this: Self) -> (*const T, RawRefGuard) {
        (this.raw.value, this.raw.guard)
    }

    /// Try to map the interior reference the reference.
    pub fn try_map<M, U: ?Sized, E>(this: Self, m: M) -> Result<Ref<'a, U>, E>
    where
        M: FnOnce(&T) -> Result<&U, E>,
    {
        let value = m(unsafe { &*this.raw.value })?;
        let guard = this.raw.guard;

        Ok(Ref {
            raw: RawRef { value, guard },
            _marker: marker::PhantomData,
        })
    }
}

impl<T: ?Sized> ops::Deref for Ref<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.raw.value }
    }
}

impl<T: ?Sized> fmt::Debug for Ref<'_, T>
where
    T: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, fmt)
    }
}

/// A raw mutable guard.
pub struct RawMutGuard {
    pub(super) access: *const Access,
}

impl Drop for RawMutGuard {
    fn drop(&mut self) {
        unsafe { (*self.access).release_exclusive() }
    }
}

/// A raw guard for exclusively borrowed values.
pub(super) struct RawMut<T: ?Sized> {
    pub(super) value: *mut T,
    pub(super) guard: RawMutGuard,
}

/// Guard for a value exclusively borrowed from a slot in the virtual machine.
///
/// These guards are necessary, since we need to guarantee certain forms of
/// access depending on what we do. Releasing the guard releases the access.
///
/// These also aid in function call integration, since they can be "arm" the
/// virtual machine to release shared guards through its unsafe functions.
///
/// See [clear][crate::Vm::clear] for more information.
pub struct Mut<'a, T: ?Sized> {
    pub(super) raw: RawMut<T>,
    pub(super) _marker: marker::PhantomData<&'a mut T>,
}

impl<'a, T: ?Sized> Mut<'a, T> {
    /// Convert into a raw pointer and associated raw access guard.
    ///
    /// # Safety
    ///
    /// The returned pointer must not outlive the associated guard, since this
    /// prevents other uses of the underlying data which is incompatible with
    /// the current.
    ///
    /// The returned pointer also must not outlive the VM that produced.
    /// Nor a call to clear the VM using [clear], since this will free up the
    /// data being referenced.
    ///
    /// [clear]: [crate::Vm::clear]
    pub unsafe fn unsafe_into_mut(this: Self) -> (*mut T, RawMutGuard) {
        (this.raw.value, this.raw.guard)
    }

    /// Map the mutable reference.
    pub fn try_map<M, U: ?Sized, E>(this: Self, m: M) -> Result<Mut<'a, U>, E>
    where
        M: FnOnce(&mut T) -> Result<&mut U, E>,
    {
        let value = m(unsafe { &mut *this.raw.value })?;
        let guard = this.raw.guard;

        Ok(Mut {
            raw: RawMut { value, guard },
            _marker: marker::PhantomData,
        })
    }
}

impl<T: ?Sized> ops::Deref for Mut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.raw.value }
    }
}

impl<T: ?Sized> ops::DerefMut for Mut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.raw.value }
    }
}

impl<T: ?Sized> fmt::Debug for Mut<'_, T>
where
    T: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, fmt)
    }
}