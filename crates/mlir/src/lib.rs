//! Rust bindings to the MLIR project.

use std::{
    ffi::{c_char, c_uint, c_void},
    fmt::{self, Formatter},
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
    ptr, slice,
    sync::Mutex,
};

use mlir_sys as ffi;
use once_cell::sync::OnceCell;
use ty::TypeSubtype;

use crate::attr::TypeAttr;

pub mod attr;
pub mod cursor;
pub mod ty;

#[doc(hidden)]
pub struct SyncContext {
    context: Context,
    mutex: Mutex<()>,
}

unsafe impl Send for SyncContext {}
unsafe impl Sync for SyncContext {}

impl SyncContext {
    /// Calls the provided closure without locking the underlying context.
    ///
    /// This is safe only for operations which the context synchronizes internally, particularly
    /// certain operations on uniqued values.
    pub fn without_mutex<F, T>(&self, f: F) -> T
    where
        F: FnOnce(ffi::MlirContext) -> T,
    {
        f(self.context.inner)
    }

    /// Calls the provided closure with the underlying context under a mutex.
    ///
    /// This is required for operations which the context does not synchronize internally.
    pub fn with_mutex<F, T>(&self, f: F) -> T
    where
        F: FnOnce(ffi::MlirContext) -> T,
    {
        let _guard = self.mutex.lock().unwrap();
        f(self.context.inner)
    }
}

static CONTEXT: OnceCell<SyncContext> = OnceCell::new();

// Defines wrappers around semantically owned MLIR values.
macro_rules! owned_types {
    ($(
        $(#[$attr:meta])*
        $v:vis struct $name:ident = $inner:path;
    )*) => {
        $(
            struct_def!($(#[$attr])* $v $name, $inner);
            raw_impls!($name, $inner);
        )*
    };
}

// Defines wrappers around semantically owned MLIR values that hold references to a `Context`.
macro_rules! semi_owned_types {
    ($(
        $(#[$attr:meta])*
        $v:vis struct $name:ident = $inner:path;
    )*) => {
        $(
            struct_def!($(#[$attr])* $v $name, $inner);
            raw_impls!($name, $inner);
        )*
    };
}

// Defines wrappers around semantically borrowed MLIR values.
macro_rules! borrowed_types {
    ($(
        $(#[$attr:meta])*
        $v:vis struct $name:ident = $inner:path;
    )*) => {
        $(
            struct_def!($(#[$attr])* #[derive(Copy, Clone)] $v $name<'a>, $inner);
            raw_impls!($name<'a>, $inner);
        )*
    };
}

macro_rules! uniqued_types {
    ($(
        $(#[$attr:meta])*
        $v:vis struct $name:ident = $inner:path;
    )*) => {
        $(
            struct_def!($(#[$attr])* #[derive(Copy, Clone)] $v $name, $inner);
            raw_impls!($name, $inner);
        )*
    };
}

macro_rules! struct_def {
    ($(#[$m:meta])* $v:vis $name:ident $(<$lt:lifetime>)? , $inner:path) => {
        $(#[$m])*
        #[repr(transparent)]
        $v struct $name $(<$lt>)? {
            #[allow(dead_code)]
            pub(crate) inner: $inner,
            $(pub (crate) phantom: PhantomData<&$lt ()>,)?
        }

        impl$(<$lt>)? private::Sealed for $name$(<$lt>)? {}
    };
}

// This macro requires two definitions because the optional lifetime doesn't appear in the
// `phantom: PhantomData` field assignment.
macro_rules! raw_impls {
    ($name:ident, $inner:path) => {
        impl $name {
            #[doc = concat!("Construct a `", stringify!($name), "` from its C API equivalent.")]
            ///
            /// # Safety
            ///
            /// Calling this constructor must not result in duplicate ownership or mutable aliasing.
            #[allow(dead_code)]
            #[inline]
            pub unsafe fn from_raw(raw: $inner) -> Option<$name> {
                if raw.ptr.is_null() {
                    return None;
                }

                Some($name { inner: raw })
            }

            #[doc = concat!("Obtain the C API equivalent of a `", stringify!($name), "`.")]
            #[allow(dead_code)]
            #[inline]
            pub fn as_raw(&self) -> $inner {
                self.inner
            }
        }
    };

    ($name:ident <$lt:lifetime> , $inner:path) => {
        impl<$lt> $name<$lt> {
            #[doc = concat!("Construct a `", stringify!($name), "` from its C API equivalent.")]
            ///
            /// # Safety
            ///
            /// Calling this constructor must not result in duplicate ownership or mutable aliasing.
            #[allow(dead_code)]
            #[inline]
            pub unsafe fn from_raw(raw: $inner) -> Option<$name<$lt>> {
                if raw.ptr.is_null() {
                    return None;
                }

                Some($name {
                    inner: raw,
                    phantom: PhantomData,
                })
            }

            #[doc = concat!("Obtain the C API equivalent of a `", stringify!($name), "`.")]
            #[allow(dead_code)]
            #[inline]
            pub fn as_raw(&self) -> $inner {
                self.inner
            }
        }
    };
}

macro_rules! impl_display {
    ($(impl$(<$lt:lifetime>)? fmt::Display for $name:ident$(<$lt2:lifetime>)? = $print_fn:path;)*) => {
        $(
            impl$(<$lt>)? fmt::Display for $name$(<$lt2>)? {
                fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                    let mut userdata = FmtUserdata::new(f);

                    unsafe {
                        $print_fn(
                            self.inner,
                            Some(fmt_callback::<Formatter<'_>>),
                            &mut userdata as *mut FmtUserdata<Formatter<'_>> as *mut c_void,
                        );
                    }

                    Ok(())
                }
            }
        )*
    };
}

macro_rules! impl_eq {
    ($(impl$(<$lt:lifetime>)? Eq for $name:ident$(<$lt2:lifetime>)? = $eq_fn:path;)*) => {
        $(
            impl$(<$lt>)? PartialEq for $name$(<$lt2>)? {
                #[inline]
                fn eq(&self, other: &Self) -> bool {
                    unsafe { $eq_fn(self.inner, other.inner) }
                }
            }

            impl$(<$lt>)? Eq for $name$(<$lt2>)? {}
        )*
    };
}

// User data object for use in MLIR formatting callbacks.
struct FmtUserdata<'fmt, W: fmt::Write> {
    w: &'fmt mut W,
    error: Option<fmt::Error>,
}

impl<'fmt, W: fmt::Write> FmtUserdata<'fmt, W> {
    fn new(w: &'fmt mut W) -> FmtUserdata<'fmt, W> {
        FmtUserdata { w, error: None }
    }
}

/// MLIR string callback for writing to a `fmt::Formatter`.
///
/// # Safety
///
/// - `userdata` must be safely upgradable to `FmtUserdata`.
pub unsafe extern "C" fn fmt_callback<W: fmt::Write>(s: ffi::MlirStringRef, userdata: *mut c_void) {
    let fmt_ptr: *mut FmtUserdata<'_, W> = userdata.cast();
    let Some(userdata): Option<&mut FmtUserdata<W>> = (unsafe { fmt_ptr.as_mut() }) else { return };
    if userdata.error.is_some() {
        return;
    }

    let bytes = unsafe { slice::from_raw_parts(s.data as *const u8, s.length) };
    let Ok(utf8) = std::str::from_utf8(bytes) else {
        userdata.error = Some(fmt::Error);
        return
    };

    let _ = write!(userdata.w, "{utf8}");
}

#[doc(hidden)]
pub fn context() -> &'static SyncContext {
    CONTEXT.get_or_init(|| SyncContext {
        context: Context::create().expect("Context creation failed."),
        mutex: Mutex::new(()),
    })
}

owned_types! {
    /// A list of operations (a basic block).
    pub struct Block = ffi::MlirBlock;
    /// A top-level context object for a collection of MLIR operations.
    struct Context = ffi::MlirContext;
    /// A handle used to register a dialect with a [`Context`].
    pub struct DialectHandle = ffi::MlirDialectHandle;
    /// A registry of dialects available to a [`Context`].
    pub struct DialectRegistry = ffi::MlirDialectRegistry;
    /// An MLIR operation.
    pub struct Operation = ffi::MlirOperation;
    /// A list of basic blocks attached to a parent operation.
    pub struct Region = ffi::MlirRegion;
}

semi_owned_types! {
    /// A collection of operations, attributes and types associated with a unique namespace.
    pub struct Dialect = ffi::MlirDialect;
    /// A top-level unit of MLIR.
    pub struct Module = ffi::MlirModule;
    /// A set of unique symbols associated with an operation.
    pub struct SymbolTable = ffi::MlirSymbolTable;
}

borrowed_types! {
    /// A reference to a [`Block`].
    pub struct BlockRef = ffi::MlirBlock;
    /// A mutable reference to a [`Block`].
    pub struct BlockMut = ffi::MlirBlock;
    /// A reference to an [`Operation`].
    pub struct OperationRef = ffi::MlirOperation;
    /// A mutable reference to an [`Operation`].
    pub struct OperationMut = ffi::MlirOperation;
    /// A reference to a [`Region`].
    pub struct RegionRef = ffi::MlirRegion;
    /// A mutable reference to a [`Region`].
    pub struct RegionMut = ffi::MlirRegion;
}

uniqued_types! {
    /// A (compile-time) constant value associated with an operation.
    pub struct Attribute = ffi::MlirAttribute;
    pub struct Identifier = ffi::MlirIdentifier;
    pub struct Location = ffi::MlirLocation;
    pub struct Type = ffi::MlirType;
    pub struct Value = ffi::MlirValue;
}

impl_eq! {
    impl Eq for Attribute = ffi::mlirAttributeEqual;
    impl Eq for Block = ffi::mlirBlockEqual;
    impl Eq for Context = ffi::mlirContextEqual;
    impl Eq for Dialect = ffi::mlirDialectEqual;
    impl Eq for Location = ffi::mlirLocationEqual;
    impl Eq for Operation = ffi::mlirOperationEqual;
    impl Eq for Region = ffi::mlirRegionEqual;
    impl Eq for Type = ffi::mlirTypeEqual;
    impl Eq for Value = ffi::mlirValueEqual;
}

impl_display! {
    impl fmt::Display for Attribute = ffi::mlirAttributePrint;
    impl fmt::Display for Location = ffi::mlirLocationPrint;
    impl fmt::Display for Operation = ffi::mlirOperationPrint;
    impl fmt::Display for Type = ffi::mlirTypePrint;
}

/// Defines methods of the form `fn(&self) -> bool`.
macro_rules! is_fns {
    (impl$(<$lt:lifetime>)? $name:ident$(<$lt2:lifetime>)? {
        $($v:vis fn $fn_name:ident = $ffi_name:path;)*
    }) => {
        impl $(<$lt>)? $name $(<$lt2>)? {
            #[inline]
            $($v fn $fn_name(&self) -> bool {
                unsafe { $ffi_name(self.inner) }
            })*
        }
    };
}

// Attribute ==================================================================

is_fns! {
    impl Attribute {
        pub fn is_affine_map = ffi::mlirAttributeIsAAffineMap;
        pub fn is_array = ffi::mlirAttributeIsAArray;
        pub fn is_bool = ffi::mlirAttributeIsABool;
        pub fn is_dictionary = ffi::mlirAttributeIsADictionary;
        pub fn is_flat_symbol_ref = ffi::mlirAttributeIsAFlatSymbolRef;
        pub fn is_float = ffi::mlirAttributeIsAFloat;
        pub fn is_integer = ffi::mlirAttributeIsAInteger;
        pub fn is_integer_set = ffi::mlirAttributeIsAIntegerSet;
        pub fn is_opaque = ffi::mlirAttributeIsAOpaque;
        pub fn is_string = ffi::mlirAttributeIsAString;
        pub fn is_symbol_ref = ffi::mlirAttributeIsASymbolRef;
        pub fn is_unit = ffi::mlirAttributeIsAUnit;
    }
}
pub(crate) use is_fns;

impl Attribute {
    #[inline]
    pub fn array(elements: &[Attribute]) -> Attribute {
        // Safety: attribute creation is synchronized internally.
        context().without_mutex(|cx| unsafe {
            Attribute::from_raw(ffi::mlirArrayAttrGet(
                cx,
                elements.len() as isize,
                elements.as_ptr() as *const _,
            ))
            .unwrap()
        })
    }

    #[inline]
    pub fn string<'a, S: Into<StringRef<'a>>>(s: S) -> Attribute {
        // Safety: attribute creation is synchronized internally.
        context().without_mutex(|cx| unsafe {
            Attribute::from_raw(ffi::mlirStringAttrGet(cx, s.into().inner)).unwrap()
        })
    }
}

impl<'a, S> From<S> for Attribute
where
    S: Into<StringRef<'a>>,
{
    #[inline]
    fn from(value: S) -> Self {
        Attribute::string(value)
    }
}

impl From<Identifier> for Attribute {
    #[inline]
    fn from(value: Identifier) -> Self {
        // Under the hood, `MlirIdentifier` is actually a `StringAttr`, so it's safe to
        // reinterpret the pointer this way.
        Attribute {
            inner: ffi::MlirAttribute {
                ptr: value.inner.ptr,
            },
        }
    }
}

impl From<Type> for Attribute {
    #[inline]
    fn from(value: Type) -> Self {
        TypeAttr::from(value).into()
    }
}

// Block ======================================================================

impl Block {
    #[inline]
    pub fn create() -> Block {
        unsafe { Block::from_raw(ffi::mlirBlockCreate(0, ptr::null(), ptr::null())).unwrap() }
    }

    #[inline]
    pub fn create_with_args(args: &[Type], locs: &[Location]) -> Block {
        assert_eq!(
            args.len(),
            locs.len(),
            "block arguments and locations arrays should be the same length"
        );

        unsafe {
            Block::from_raw(ffi::mlirBlockCreate(
                args.len() as isize,
                args.as_ptr() as *const ffi::MlirType,
                locs.as_ptr() as *const ffi::MlirLocation,
            ))
            .unwrap()
        }
    }

    #[inline]
    pub fn add_argument(&mut self, ty: Type, loc: Location) {
        unsafe {
            ffi::mlirBlockAddArgument(self.inner, ty.inner, loc.inner);
        }
    }

    #[inline]
    pub fn terminator(&self) -> Option<OperationRef> {
        unsafe { OperationRef::from_raw(ffi::mlirBlockGetTerminator(self.inner)) }
    }

    /// Returns the closest surrounding operation that contains this block.
    ///
    /// Returns `None` if this block is unlinked.
    #[inline]
    pub fn parent_operation(&self) -> Option<OperationRef> {
        unsafe { OperationRef::from_raw(ffi::mlirBlockGetParentOperation(self.inner)) }
    }
}

impl Deref for BlockRef<'_> {
    type Target = Block;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const BlockRef as *const Block) }
    }
}

impl Deref for BlockMut<'_> {
    type Target = Block;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const BlockMut as *const Block) }
    }
}

impl DerefMut for BlockMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut BlockMut as *mut Block) }
    }
}

impl Context {
    /// Creates an MLIR context.
    pub fn create() -> Option<Context> {
        let ctx = unsafe { ffi::mlirContextCreate() };

        if ctx.ptr.is_null() {
            return None;
        }

        Some(Context { inner: ctx })
    }

    /// Registers all dialects in `registry` with this MLIR context.
    pub fn append_dialect_registry(&mut self, registry: &DialectRegistry) {
        unsafe { ffi::mlirContextAppendDialectRegistry(self.inner, registry.inner) }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { ffi::mlirContextDestroy(self.inner) }
    }
}

// Dialect ====================================================================

impl Dialect {
    pub fn namespace(&self) -> StringRef {
        unsafe { StringRef::from_raw(ffi::mlirDialectGetNamespace(self.inner)) }
    }
}

// DialectHandle ==============================================================

impl DialectHandle {
    pub fn register_dialect(&self) {
        context().with_mutex(|cx| unsafe { ffi::mlirDialectHandleRegisterDialect(self.inner, cx) })
    }

    pub fn load_dialect(&self) -> Option<Dialect> {
        context().with_mutex(|cx| unsafe {
            Dialect::from_raw(ffi::mlirDialectHandleLoadDialect(self.inner, cx))
        })
    }

    pub fn namespace(&self) -> StringRef {
        unsafe {
            let s = ffi::mlirDialectHandleGetNamespace(self.inner);
            StringRef::from_raw(s)
        }
    }
}

// DialectRegistry ============================================================

impl Drop for DialectRegistry {
    fn drop(&mut self) {
        unsafe { ffi::mlirDialectRegistryDestroy(self.inner) }
    }
}

impl DialectRegistry {
    pub fn create() -> Option<DialectRegistry> {
        let reg = unsafe { ffi::mlirDialectRegistryCreate() };

        if reg.ptr.is_null() {
            return None;
        }

        Some(DialectRegistry { inner: reg })
    }
}

// Identifier =================================================================

impl Identifier {
    pub fn get<'a, S: Into<StringRef<'a>>>(value: S) -> Identifier {
        context().without_mutex(|cx| unsafe {
            Identifier::from_raw(ffi::mlirIdentifierGet(cx, value.into().inner)).unwrap()
        })
    }

    pub fn value(&self) -> StringRef {
        unsafe { StringRef::from_raw(ffi::mlirIdentifierStr(self.inner)) }
    }
}

impl<'a, S> From<S> for Identifier
where
    S: Into<StringRef<'a>>,
{
    fn from(value: S) -> Self {
        Identifier::get(value)
    }
}

// Location ===================================================================

impl Location {
    pub fn call_site(callee: Location, caller: Location) -> Location {
        // `UnknownLoc` is cached in `Context`, so it should always be present.
        unsafe {
            Location::from_raw(ffi::mlirLocationCallSiteGet(
                callee.as_raw(),
                caller.as_raw(),
            ))
            .unwrap()
        }
    }

    pub fn file_line_col<'a, S: Into<StringRef<'a>>>(filename: S, line: u32, col: u32) -> Location {
        #![allow(clippy::useless_conversion)]

        // These are no-ops on almost every platform, but it's possible for c_uint to be a u16.
        let line: c_uint = line.try_into().unwrap();
        let col: c_uint = col.try_into().unwrap();

        // Safety: Locations are uniqued and therefore synchronized.
        context().without_mutex(|cx| unsafe {
            Location::from_raw(ffi::mlirLocationFileLineColGet(
                cx,
                filename.into().as_raw(),
                line,
                col,
            ))
            .unwrap()
        })
    }

    pub fn fused<A: Into<Attribute>>(locations: &[Location], metadata: Attribute) -> Location {
        // Safety: Locations are uniqued and therefore synchronized.
        context().without_mutex(|cx| unsafe {
            // `UnknownLoc` is cached in `Context`, so it should always be present.
            Location::from_raw(ffi::mlirLocationFusedGet(
                cx,
                locations.len() as isize,
                // Location is #[repr(transparent)] around `MlirLocation`.
                locations.as_ptr() as *const ffi::MlirLocation,
                metadata.as_raw(),
            ))
            .unwrap()
        })
    }

    pub fn unknown() -> Location {
        // Safety: Locations are uniqued and therefore synchronized.
        context().without_mutex(|cx| unsafe {
            // `UnknownLoc` is cached in `Context`, so it should always be present.
            Location::from_raw(ffi::mlirLocationUnknownGet(cx))
                .expect("unexpected MLIR error: UnknownLoc should be non-null")
        })
    }
}

// Module =====================================================================

impl Module {
    pub fn create_empty(location: Location) -> Module {
        unsafe {
            Module::from_raw(ffi::mlirModuleCreateEmpty(location.inner))
                .expect("unexpected MLIR error: empty Module should be non-null")
        }
    }

    pub fn create_parse<'src, S: Into<StringRef<'src>>>(module: S) -> Option<Module> {
        context().with_mutex(|cx| unsafe {
            Module::from_raw(ffi::mlirModuleCreateParse(cx, module.into().inner))
        })
    }

    pub fn body(&self) -> Block {
        unsafe {
            Block::from_raw(ffi::mlirModuleGetBody(self.inner))
                .expect("unexpected MLIR error: Module body should be non-null")
        }
    }
}

// NamedAttribute =============================================================

#[derive(Copy, Clone)]
#[repr(C)]
pub struct NamedAttribute {
    pub name: Identifier,
    pub attribute: Attribute,
}

impl NamedAttribute {
    pub fn get<I, A>(name: I, attribute: A) -> NamedAttribute
    where
        I: Into<Identifier>,
        A: Into<Attribute>,
    {
        NamedAttribute {
            name: name.into(),
            attribute: attribute.into(),
        }
    }

    pub(crate) fn as_raw(&self) -> ffi::MlirNamedAttribute {
        ffi::MlirNamedAttribute {
            name: self.name.inner,
            attribute: self.attribute.inner,
        }
    }
}

// Operation ==================================================================

impl Operation {
    pub fn create(mut state: OperationState<'_>) -> Option<Operation> {
        let op = unsafe { ffi::mlirOperationCreate(&mut state.inner as *mut _) };

        if op.ptr.is_null() {
            return None;
        }

        Some(Operation { inner: op })
    }
}

impl Deref for OperationRef<'_> {
    type Target = Operation;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const OperationRef as *const Operation) }
    }
}

impl Deref for OperationMut<'_> {
    type Target = Operation;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const OperationMut as *const Operation) }
    }
}

impl DerefMut for OperationMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut OperationMut as *mut Operation) }
    }
}

// OperationState =============================================================

#[repr(transparent)]
pub struct OperationState<'name> {
    inner: ffi::MlirOperationState,
    phantom: PhantomData<&'name ()>,
}

impl<'name> OperationState<'name> {
    pub fn get<S: Into<StringRef<'name>>>(name: S, loc: Location) -> OperationState<'name> {
        let sref = name.into();
        OperationState {
            inner: unsafe { ffi::mlirOperationStateGet(sref.inner, loc.inner) },
            phantom: PhantomData,
        }
    }

    pub fn add_attribute<I, A>(&mut self, ident: I, attribute: A)
    where
        I: Into<Identifier>,
        A: Into<Attribute>,
    {
        self.add_attributes(&[NamedAttribute::get(ident.into(), attribute.into())]);
    }

    pub fn add_attributes(&mut self, attributes: &[NamedAttribute]) {
        unsafe {
            ffi::mlirOperationStateAddAttributes(
                &mut self.inner as *mut _,
                attributes.len() as isize,
                attributes.as_ptr() as *const _,
            );
        }
    }

    pub fn add_results(&mut self, results: &[Type]) {
        unsafe {
            ffi::mlirOperationStateAddResults(
                &mut self.inner as *mut _,
                results.len() as isize,
                results.as_ptr() as *const _,
            );
        }
    }

    pub fn add_operands(&mut self, operands: &[Value]) {
        unsafe {
            ffi::mlirOperationStateAddOperands(
                &mut self.inner as *mut _,
                operands.len() as isize,
                operands.as_ptr() as *const _,
            );
        }
    }

    pub fn add_region(&mut self, region: Region) -> RegionMut {
        // Don't drop the region.
        let region = ManuallyDrop::new(region);

        unsafe {
            ffi::mlirOperationStateAddOwnedRegions(
                &mut self.inner as *mut _,
                1,
                &region.inner as *const _,
            );

            RegionMut::from_raw(region.inner).unwrap()
        }
    }

    pub fn add_regions(&mut self, regions: Vec<Region>) {
        // Decompose the vector into its raw parts.
        let ptr: *mut ManuallyDrop<Region> = regions.as_ptr() as *mut _;
        let len = regions.len();
        let cap = regions.capacity();

        // Don't drop the decomposed vector.
        let _ = ManuallyDrop::new(regions);

        // Convert to Vec<ManuallyDrop<Region>>.
        let regions = unsafe { Vec::from_raw_parts(ptr, len, cap) };

        unsafe {
            ffi::mlirOperationStateAddOwnedRegions(
                &mut self.inner as *mut _,
                regions.len() as isize,
                regions.as_ptr() as *const _,
            );
        }

        // The new vector drops without dropping its elements.
    }
}

// Region =====================================================================

impl Region {
    pub fn create() -> Region {
        unsafe { Region::from_raw(ffi::mlirRegionCreate()).unwrap() }
    }

    pub fn append_block(&mut self, block: Block) -> BlockMut {
        let block = ManuallyDrop::new(block);

        unsafe {
            ffi::mlirRegionAppendOwnedBlock(self.inner, block.inner);
            BlockMut::from_raw(block.inner).unwrap()
        }
    }
}

impl Deref for RegionRef<'_> {
    type Target = Region;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const RegionRef as *const Region) }
    }
}

impl Deref for RegionMut<'_> {
    type Target = Region;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const RegionMut as *const Region) }
    }
}

impl DerefMut for RegionMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut RegionMut as *mut Region) }
    }
}

// StringRef ==================================================================

#[derive(Copy, Clone)]
pub struct StringRef<'a> {
    pub(crate) inner: ffi::MlirStringRef,
    pub(crate) phantom: PhantomData<&'a c_char>,
}

impl<'a> From<&'a str> for StringRef<'a> {
    fn from(value: &'a str) -> Self {
        let nul_term = value.as_bytes().last() == Some(&b'\0');

        StringRef {
            inner: ffi::MlirStringRef {
                data: value.as_ptr() as *const i8,
                length: value.len() - nul_term as usize,
            },
            phantom: PhantomData,
        }
    }
}

impl<'a> From<&'a String> for StringRef<'a> {
    fn from(value: &'a String) -> Self {
        Self::from(value.as_str())
    }
}

impl<'a> StringRef<'a> {
    pub(crate) unsafe fn from_raw(s: ffi::MlirStringRef) -> StringRef<'a> {
        StringRef {
            inner: s,
            phantom: PhantomData,
        }
    }

    pub(crate) fn as_raw(&self) -> ffi::MlirStringRef {
        self.inner
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.inner.data as *const u8, self.inner.length) }
    }

    pub fn to_str(&self) -> Option<&str> {
        std::str::from_utf8(self.as_bytes()).ok()
    }
}

// SymbolTable ================================================================

impl SymbolTable {
    pub fn symbol_attribute_name() -> StringRef<'static> {
        unsafe { StringRef::from_raw(ffi::mlirSymbolTableGetSymbolAttributeName()) }
    }

    pub fn visibility_attribute_name() -> StringRef<'static> {
        unsafe { StringRef::from_raw(ffi::mlirSymbolTableGetVisibilityAttributeName()) }
    }
}

// Type =======================================================================

impl Type {
    pub fn downcast<T: TypeSubtype>(self) -> Result<T, Self> {
        T::downcast_from(self)
    }
}

mod private {
    pub trait Sealed {}
}
