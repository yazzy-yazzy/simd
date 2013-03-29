#ifndef __ICC
#error please use the Intel C/C++ Compiler.
#endif // __ICC

#include <cassert>

#include <ia32intrin.h>	// IA32
#include <mmintrin.h>	// MMX
#include <xmmintrin.h>	// SSE
#include <emmintrin.h>	// SSE2
#include <pmmintrin.h>	// SSE3
#include <tmmintrin.h>	// SSSE3
#include <nmmintrin.h>	// SSE4
#include <smmintrin.h>	// SSE4

typedef __m128i ( *load_si128_t )( __m128i const*p );
typedef void ( *store_si128_t )( __m128i *p, __m128i b );
typedef void ( *store_si32_t )( int *p, int b );

const unsigned char TRANSPARENT = 0x80;

enum
{
	ALIGN_NONE = 0,
	ALIGN_BYTE = 1,
	ALIGN_WORD = 2,
	ALIGN_DWORD = 4,
	ALIGN_QWORD = 8,
	ALIGN_DQWORD = 16
};

static void memset_normal( void *src, int c, size_t n );
static void memset_dqword( void *src, int c, size_t n, int2type< ALIGN_NONE > );
static void memset_dqword( void *src, int c, size_t n, int2type< ALIGN_DQWORD > );

static void memcpy_normal( void *dst, void *src, size_t n );
static void memcpy_dqword( void *dst, void *src, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_NONE > );
static void memcpy_dqword( void *dst, void *src, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_DQWORD > );
static void memcpy_dqword( void *dst, void *src, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_DQWORD > );
static void memcpy_dqword( void *dst, void *src, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_NONE > );

static int memccmp_dqword( void *src, int c, size_t n, int2type< ALIGN_DQWORD > );
static int memccmp_dqword( void *src, int c, size_t n, int2type< ALIGN_NONE > );
static int memccmp_normal( void *src, int c, size_t n );

static void maskset_normal( void *dst, int c, void *srcmask, size_t n );
static void maskset_64byte( void *dst, int c, void *srcmask, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_NONE > );
static void maskset_64byte( void *dst, int c, void *srcmask, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_DQWORD > );
static void maskset_64byte( void *dst, int c, void *srcmask, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_DQWORD > );
static void maskset_64byte( void *dst, int c, void *srcmask, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_NONE > );

static void maskcopy_normal( void *dst, void *src, void *srcmask, size_t n );
static void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_NONE >, int2type< ALIGN_NONE > );
static void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_NONE >, int2type< ALIGN_DQWORD > );
static void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_DQWORD >, int2type< ALIGN_NONE > );
static void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_DQWORD >, int2type< ALIGN_DQWORD > );
static void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_NONE >, int2type< ALIGN_NONE > );
static void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_NONE >, int2type< ALIGN_DQWORD > );
static void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_DQWORD >, int2type< ALIGN_NONE > );
static void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_DQWORD >, int2type< ALIGN_DQWORD > );

static int maskccmp_dqword( void *src, size_t n, int2type< ALIGN_DQWORD > );
static int maskccmp_dqword( void *src, size_t n, int2type< ALIGN_NONE > );
static int maskccmp_normal( void *src, size_t n );



static inline bool is_dqword_aligned( void *ptr )
{
	return ( reinterpret_cast< unsigned int >( ptr ) & 0x0000000F ? false : true );
}


template < store_si128_t store_si128 >
inline void memset_dqword( void *src, int c, size_t n )
{
	assert( 16 <= n );

	__m128i *psrc = static_cast< __m128i * >( src );
	const __m128i v = _mm_set1_epi8( c );
	size_t i = n >> 4;	// n / 16

	do store_si128( psrc++, v );
	while ( --i );
}

template < load_si128_t load_si128, store_si128_t store_si128 >
inline void memcpy_dqword( void *dst, void *src, size_t n )
{
	assert( 16 <= n );

	__m128i *psrc = static_cast< __m128i * >( src );
	__m128i *pdst = static_cast< __m128i * >( dst );
	size_t i = n >> 4;	// n / 16

	do store_si128( pdst++, load_si128( psrc++ ) );
	while ( --i );
}

template < load_si128_t load_si128 >
inline int memccmp_dqword( void *src, int c, size_t n )
{
	assert( 16 <= n );

	const __m128i mask = _mm_set1_epi8( c );
	__m128i *psrc = static_cast< __m128i * >( src );
	size_t i = n >> 4;	// n / 16

	do
	{
		if ( 0x0000FFFF != _mm_movemask_epi8( _mm_cmpeq_epi8( load_si128( psrc++ ), mask ) ) )
			break;
	}
	while ( --i );

	return ( i ? 1 : 0 );
}

template < load_si128_t load_si128 >
inline int maskccmp_dqword( void *src, size_t n )
{
	assert( 16 <= n );

	const __m128i mask = _mm_set1_epi8( TRANSPARENT );
	__m128i *psrc = static_cast< __m128i * >( src );
	size_t i = n >> 4;	// n / 16

	do
	{
		if ( 0x0000FFFF != _mm_movemask_epi8( _mm_cmpeq_epi8( _mm_and_si128( load_si128( psrc++ ), mask ), mask ) ) )
			break;
	}
	while ( --i );

	return ( i ? 1 : 0 );
}

template < load_si128_t load_si128_src, load_si128_t load_si128_tag, load_si128_t load_si128_dst, store_si128_t store_si128 >
inline void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n )
{
	assert( 64 <= n );

	__m128i *psrc, *pdst, *psrcmask;
	__m128i mask[2];
	int mask_msb;

	size_t i = n >> 6;	// n / 64;

	pdst = static_cast< __m128i * >( dst );
	psrc = static_cast< __m128i * >( src );
	psrcmask = static_cast< __m128i * >( srcmask );

	const __m128i mask_1 = _mm_set1_epi32( 0xFFFFFFFF );
	const __m128i mask_transparent = _mm_set1_epi8( TRANSPARENT );

	const size_t distance = 64*4;

	do
	{
		_mm_prefetch( reinterpret_cast< char * >( psrcmask ) + distance, _MM_HINT_T0 );

		mask[0] = _mm_xor_si128( _mm_cmpeq_epi8( _mm_and_si128( load_si128_tag( psrcmask ), mask_transparent ), mask_transparent ), mask_1 );
		mask[1] = _mm_xor_si128( _mm_cmpeq_epi8( _mm_and_si128( load_si128_tag( psrcmask+1 ), mask_transparent ), mask_transparent ), mask_1 );
		mask_msb = ( _mm_movemask_epi8( mask[1] ) << 16 ) | _mm_movemask_epi8( mask[0] );

		if ( mask_msb )
		{
			_mm_prefetch( reinterpret_cast< char * >( psrc ) + distance, _MM_HINT_NTA );

			if ( 0xFFFFFFFF == mask_msb )
			{
				store_si128( pdst, load_si128_src( psrc ) );
				store_si128( pdst+1, load_si128_src( psrc+1 ) );
			}
			else if ( 0x0000FFFF == mask_msb )
			{
				store_si128( pdst, load_si128_src( psrc ) );
			}
			else if ( 0xFFFF0000 == mask_msb )
			{
				store_si128( pdst+1, load_si128_src( psrc+1 ) );
			}
			else
			{
				_mm_prefetch( reinterpret_cast< char * >( pdst ) + distance, _MM_HINT_NTA );

				store_si128( pdst,
					_mm_adds_epu8(
						_mm_and_si128( load_si128_src( psrc ), mask[0] ),
						_mm_subs_epu8( load_si128_dst( pdst ), mask[0] )
					)
				);

				store_si128( pdst+1,
					_mm_adds_epu8(
						_mm_and_si128( load_si128_src( psrc+1 ), mask[1] ),
						_mm_subs_epu8( load_si128_dst( pdst+1 ), mask[1] )
					)
				);
			}
		}

		mask[0] = _mm_xor_si128( _mm_cmpeq_epi8( _mm_and_si128( load_si128_tag( psrcmask+2 ), mask_transparent ), mask_transparent ), mask_1 );
		mask[1] = _mm_xor_si128( _mm_cmpeq_epi8( _mm_and_si128( load_si128_tag( psrcmask+3 ), mask_transparent ), mask_transparent ), mask_1 );
		mask_msb = ( _mm_movemask_epi8( mask[1] ) << 16 ) | _mm_movemask_epi8( mask[0] );

		if ( mask_msb )
		{
			if ( 0xFFFFFFFF == mask_msb )
			{
				store_si128( pdst+2, load_si128_src( psrc+2 ) );
				store_si128( pdst+3, load_si128_src( psrc+3 ) );
			}
			else if ( 0x0000FFFF == mask_msb )
			{
				store_si128( pdst+2, load_si128_src( psrc+2 ) );
			}
			else if ( 0xFFFF0000 == mask_msb )
			{
				store_si128( pdst+3, load_si128_src( psrc+3 ) );
			}
			else
			{
				store_si128( pdst+2,
					_mm_adds_epu8(
						_mm_and_si128( load_si128_src( psrc+2 ), mask[0] ),
						_mm_subs_epu8( load_si128_dst( pdst+2 ), mask[0] )
					)
				);

				store_si128( pdst+3,
					_mm_adds_epu8(
						_mm_and_si128( load_si128_src( psrc+3 ), mask[1] ),
						_mm_subs_epu8( load_si128_dst( pdst+3 ), mask[1] )
					)
				);
			}
		}

		psrc += 4;
		pdst += 4;
		psrcmask += 4;
	}
	while ( --i );
}

template < load_si128_t load_si128_src, load_si128_t load_si128_dst, store_si128_t store_si128 >
inline void maskset_64byte( void *dst, int c, void *srcmask, size_t n )
{
	assert( 64 <= n );

	__m128i *pdst, *psrcmask;
	__m128i mask[2];
	int mask_msb;

	size_t i = n >> 6;	// n / 64;

	pdst = static_cast< __m128i * >( dst );
	psrcmask = static_cast< __m128i * >( srcmask );

	const __m128i v = _mm_set1_epi8( c );
	const __m128i mask_1 = _mm_set1_epi32( 0xFFFFFFFF );
	const __m128i mask_transparent = _mm_set1_epi8( TRANSPARENT );

	const size_t distance = 64*4;

	do
	{
		_mm_prefetch( reinterpret_cast< char * >( psrcmask ) + distance, _MM_HINT_T0 );

		mask[0] = _mm_xor_si128( _mm_cmpeq_epi8( _mm_and_si128( load_si128_src( psrcmask ), mask_transparent ), mask_transparent ), mask_1 );
		mask[1] = _mm_xor_si128( _mm_cmpeq_epi8( _mm_and_si128( load_si128_src( psrcmask+1 ), mask_transparent ), mask_transparent ), mask_1 );
		mask_msb = ( _mm_movemask_epi8( mask[1] ) << 16 ) | _mm_movemask_epi8( mask[0] );

		if ( mask_msb )
		{
			if ( 0xFFFFFFFF == mask_msb )
			{
				store_si128( pdst, v );
				store_si128( pdst+1, v );
			}
			else if ( 0x0000FFFF == mask_msb )
			{
				store_si128( pdst, v );
			}
			else if ( 0xFFFF0000 == mask_msb )
			{
				store_si128( pdst+1, v );
			}
			else
			{
				_mm_prefetch( reinterpret_cast< char * >( pdst ) + distance, _MM_HINT_NTA );

				store_si128( pdst,
					_mm_adds_epu8(
						_mm_and_si128( v, mask[0] ),
						_mm_subs_epu8( load_si128_dst( pdst ), mask[0] )
					)
				);

				store_si128( pdst+1,
					_mm_adds_epu8(
						_mm_and_si128( v, mask[1] ),
						_mm_subs_epu8( load_si128_dst( pdst+1 ), mask[1] )
					)
				);
			}
		}

		mask[0] = _mm_xor_si128( _mm_cmpeq_epi8( _mm_and_si128( load_si128_src( psrcmask+2 ), mask_transparent ), mask_transparent ), mask_1 );
		mask[1] = _mm_xor_si128( _mm_cmpeq_epi8( _mm_and_si128( load_si128_src( psrcmask+3 ), mask_transparent ), mask_transparent ), mask_1 );
		mask_msb = ( _mm_movemask_epi8( mask[1] ) << 16 ) | _mm_movemask_epi8( mask[0] );

		if ( mask_msb )
		{
			if ( 0xFFFFFFFF == mask_msb )
			{
				store_si128( pdst+2, v );
				store_si128( pdst+3, v );
			}
			else if ( 0x0000FFFF == mask_msb )
			{
				store_si128( pdst+2, v );
			}
			else if ( 0xFFFF0000 == mask_msb )
			{
				store_si128( pdst+3, v );
			}
			else
			{
				store_si128( pdst+2,
					_mm_adds_epu8(
						_mm_and_si128( v, mask[0] ),
						_mm_subs_epu8( load_si128_dst( pdst+2 ), mask[0] )
					)
				);

				store_si128( pdst+3,
					_mm_adds_epu8(
						_mm_and_si128( v, mask[1] ),
						_mm_subs_epu8( load_si128_dst( pdst+3 ), mask[1] )
					)
				);
			}
		}

		pdst += 4;
		psrcmask += 4;
	}
	while ( --i );
}



inline void memset_normal( void *src, int c, size_t n )
{
	if ( 4 <= n )
	{
		int *psrc = static_cast< int * >( src );
		const int v = ( c << 24 ) | ( c << 16 ) | ( c << 8 ) | c;
		size_t i = n >> 2;	// n / 4;

		do *psrc++ = v;
		while ( --i );

		size_t remain = n % 4;

		if ( remain )
		{
			unsigned char *psrc = static_cast< unsigned char * >( src ) + ( n - remain );

			do *psrc++ = c;
			while ( --remain );
		}
	}
	else	// 4 > n
	{
		unsigned char *psrc = static_cast< unsigned char * >( src );

		do *psrc++ = c;
		while ( --n );
	}
}

inline void memset_dqword( void *src, int c, size_t n, int2type< ALIGN_NONE > )
{
	assert( 16 <= n );

	memset_dqword< _mm_storeu_si128 >( src, c, n );
}

inline void memset_dqword( void *src, int c, size_t n, int2type< ALIGN_DQWORD > )
{
	assert( 16 <= n );
	assert( is_dqword_aligned( src ) );

	memset_dqword< _mm_stream_si128 >( src, c, n );
}



inline void memcpy_normal( void *dst, void *src, size_t n )
{
	if ( 4 <= n )
	{
		int *pdst = static_cast< int * >( dst );
		int *psrc = static_cast< int * >( src );
		size_t i = n >> 2;	// n / 4;

		do *pdst++ = *psrc++;
		while ( --i );

		size_t remain = n % 4;

		if ( remain )
		{
			unsigned char *pdst = static_cast< unsigned char * >( dst ) + ( n - remain );
			unsigned char *psrc = static_cast< unsigned char * >( src ) + ( n - remain );

			do *pdst++ = *psrc++;
			while ( --remain );
		}
	}
	else	// 4 > n
	{
		unsigned char *pdst = static_cast< unsigned char * >( dst );
		unsigned char *psrc = static_cast< unsigned char * >( src );

		do *pdst++ = *psrc++;
		while ( --n );
	}
}

inline void memcpy_dqword( void *dst, void *src, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_NONE > )
{
	assert( 16 <= n );

	memcpy_dqword< _mm_lddqu_si128, _mm_storeu_si128 >( dst, src, n );
}

inline void memcpy_dqword( void *dst, void *src, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_DQWORD > )
{
	assert( 16 <= n );
	assert( is_dqword_aligned( dst ) );
	assert( is_dqword_aligned( src ) );

	memcpy_dqword< _mm_load_si128, _mm_stream_si128 >( dst, src, n );
}

inline void memcpy_dqword( void *dst, void *src, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_DQWORD > )
{
	assert( 16 <= n );
	assert( is_dqword_aligned( dst ) );

	memcpy_dqword< _mm_lddqu_si128, _mm_stream_si128 >( dst, src, n );
}

inline void memcpy_dqword( void *dst, void *src, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_NONE > )
{
	assert( 16 <= n );
	assert( is_dqword_aligned( src ) );

	memcpy_dqword< _mm_load_si128, _mm_storeu_si128 >( dst, src, n );
}



inline int memccmp_dqword( void *src, int c, size_t n, int2type< ALIGN_DQWORD > )
{
	assert( 16 <= n );
	assert( is_dqword_aligned( src ) );

	return memccmp_dqword< _mm_load_si128 >( src, c, n );
}

inline int memccmp_dqword( void *src, int c, size_t n, int2type< ALIGN_NONE > )
{
	assert( 16 <= n );

	return memccmp_dqword< _mm_lddqu_si128 >( src, c, n );
}

inline int memccmp_normal( void *src, int c, size_t n )
{
	int result = 0;

	if ( 4 <= n )
	{
		const int mask = ( c << 24 ) | ( c << 16 ) | ( c << 8 ) | c;
		int *psrc = static_cast< int * >( src );
		size_t i = n >> 2;	// n / 4;

		do if ( mask != *psrc++ ) break;
		while ( --i );

		result = ( i ? 1 : 0 );

		if ( 0 == result )
		{
			size_t remain = n % 4;

			if ( remain )
			{
				unsigned char *psrc = static_cast< unsigned char * >( src ) + ( n - remain );

				do if ( c != *psrc++ ) break;
				while ( --remain );

				result = ( remain ? 1 : 0 );
			}
		}
	}
	else	// 4 > n
	{
		unsigned char *psrc = static_cast< unsigned char * >( src );

		do if ( c != *psrc++ ) break;
		while ( --n );

		result = ( n ? 1 : 0 );
	}

	return result;
}



inline int maskccmp_dqword( void *src, size_t n, int2type< ALIGN_DQWORD > )
{
	assert( 16 <= n );
	assert( is_dqword_aligned( src ) );

	return maskccmp_dqword< _mm_load_si128 >( src, n );
}

inline int maskccmp_dqword( void *src, size_t n, int2type< ALIGN_NONE > )
{
	assert( 16 <= n );

	return maskccmp_dqword< _mm_lddqu_si128 >( src, n );
}

inline int maskccmp_normal( void *src, size_t n )
{
	int result = 0;

	if ( 4 <= n )
	{
		const int mask = ( TRANSPARENT << 24 ) | ( TRANSPARENT << 16 ) | ( TRANSPARENT << 8 ) | TRANSPARENT;
		int *psrc = static_cast< int * >( src );
		size_t i = n >> 2;	// n / 4;

		do if ( ( *psrc++ & mask ) != mask ) break;
		while ( --i );

		result = ( i ? 1 : 0 );

		if ( 0 == result )
		{
			size_t remain = n % 4;

			if ( remain )
			{
				unsigned char *psrc = static_cast< unsigned char * >( src ) + ( n - remain );

				do if ( ( *psrc++ & TRANSPARENT ) != TRANSPARENT ) break;
				while ( --remain );

				result = ( remain ? 1 : 0 );
			}
		}
	}
	else	// 4 > n
	{
		unsigned char *psrc = static_cast< unsigned char * >( src );

		do if ( ( *psrc++ & TRANSPARENT ) != TRANSPARENT ) break;
		while ( --n );

		result = ( n ? 1 : 0 );
	}

	return result;
}



inline void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_NONE >, int2type< ALIGN_NONE > )
{
	assert( 64 <= n );

	maskcopy_64byte< _mm_lddqu_si128, _mm_lddqu_si128, _mm_lddqu_si128, _mm_storeu_si128 >( dst, src, srcmask, n );
}

inline void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_NONE >, int2type< ALIGN_DQWORD > )
{
	assert( 64 <= n );
	assert( is_dqword_aligned( dst ) );

	maskcopy_64byte< _mm_lddqu_si128, _mm_lddqu_si128, _mm_load_si128, _mm_store_si128 >( dst, src, srcmask, n );
}

inline void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_DQWORD >, int2type< ALIGN_NONE > )
{
	assert( 64 <= n );
	assert( is_dqword_aligned( srcmask ) );

	maskcopy_64byte< _mm_lddqu_si128, _mm_load_si128, _mm_lddqu_si128, _mm_storeu_si128 >( dst, src, srcmask, n );
}

inline void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_DQWORD >, int2type< ALIGN_DQWORD > )
{
	assert( 64 <= n );
	assert( is_dqword_aligned( srcmask ) );
	assert( is_dqword_aligned( dst ) );

	maskcopy_64byte< _mm_lddqu_si128, _mm_load_si128, _mm_load_si128, _mm_store_si128 >( dst, src, srcmask, n );
}

inline void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_NONE >, int2type< ALIGN_NONE > )
{
	assert( 64 <= n );
	assert( is_dqword_aligned( src ) );

	maskcopy_64byte< _mm_load_si128, _mm_lddqu_si128, _mm_lddqu_si128, _mm_storeu_si128 >( dst, src, srcmask, n );
}

inline void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_NONE >, int2type< ALIGN_DQWORD > )
{
	assert( 64 <= n );
	assert( is_dqword_aligned( src ) );
	assert( is_dqword_aligned( dst ) );

	maskcopy_64byte< _mm_load_si128, _mm_lddqu_si128, _mm_load_si128, _mm_store_si128 >( dst, src, srcmask, n );
}

inline void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_DQWORD >, int2type< ALIGN_NONE > )
{
	assert( 64 <= n );
	assert( is_dqword_aligned( src ) );
	assert( is_dqword_aligned( srcmask ) );

	maskcopy_64byte< _mm_load_si128, _mm_load_si128, _mm_lddqu_si128, _mm_storeu_si128 >( dst, src, srcmask, n );
}

inline void maskcopy_64byte( void *dst, void *src, void *srcmask, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_DQWORD >, int2type< ALIGN_DQWORD > )
{
	assert( 64 <= n );
	assert( is_dqword_aligned( dst ) );
	assert( is_dqword_aligned( src ) );
	assert( is_dqword_aligned( srcmask ) );

	maskcopy_64byte< _mm_load_si128, _mm_load_si128, _mm_load_si128, _mm_store_si128 >( dst, src, srcmask, n );
}

inline void maskcopy_normal( void *dst, void *src, void *srcmask, size_t n )
{
	unsigned char *psrc, *pdst, *psrcmask;

	pdst = static_cast< unsigned char * >( dst );
	psrc = static_cast< unsigned char * >( src );
	psrcmask = static_cast< unsigned char * >( srcmask );

	do
	{
		if ( ( *psrcmask & TRANSPARENT ) != TRANSPARENT )
			*pdst = *psrc;

		++psrc;
		++pdst;
		++psrcmask;
	}
	while ( --n );
}



inline void maskset_64byte( void *dst, int c, void *srcmask, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_DQWORD > )
{
	assert( 64 <= n );
	assert( is_dqword_aligned( dst ) );
	assert( is_dqword_aligned( srcmask ) );

	maskset_64byte< _mm_load_si128, _mm_load_si128, _mm_store_si128 >( dst, c, srcmask, n );
}

inline void maskset_64byte( void *dst, int c, void *srcmask, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_NONE > )
{
	assert( 64 <= n );

	maskset_64byte< _mm_lddqu_si128, _mm_lddqu_si128, _mm_storeu_si128 >( dst, c, srcmask, n );
}

inline void maskset_64byte( void *dst, int c, void *srcmask, size_t n, int2type< ALIGN_NONE >, int2type< ALIGN_DQWORD > )
{
	assert( 64 <= n );
	assert( is_dqword_aligned( dst ) );

	maskset_64byte< _mm_lddqu_si128, _mm_load_si128, _mm_store_si128 >( dst, c, srcmask, n );
}

inline void maskset_64byte( void *dst, int c, void *srcmask, size_t n, int2type< ALIGN_DQWORD >, int2type< ALIGN_NONE > )
{
	assert( 64 <= n );
	assert( is_dqword_aligned( srcmask ) );

	maskset_64byte< _mm_load_si128, _mm_lddqu_si128, _mm_storeu_si128 >( dst, c, srcmask, n );
}

inline void maskset_normal( void *dst, int c, void *srcmask, size_t n )
{
	unsigned char *pdst, *psrcmask;

	pdst = static_cast< unsigned char * >( dst );
	psrcmask = static_cast< unsigned char * >( srcmask );

	do
	{
		if ( ( *psrcmask & TRANSPARENT ) != TRANSPARENT )
			*pdst = c;

		++pdst;
		++psrcmask;
	}
	while ( --n );
}



/**
 * @brief fast_memset
 *
 * @note
 *   dst[n] = c
 */
void *fast_memset( void *src, int c, size_t n )
{
	if ( 16 <= n )
	{
		const bool src_aligned = is_dqword_aligned( src );
		size_t n2 = n;

		if ( src_aligned )
		{
			memset_dqword( src, c, n, int2type< ALIGN_DQWORD >() );
		}
		else
		{
			size_t remain_top = 16 - ( reinterpret_cast< unsigned int >( src ) % 16 );

			if ( 16 > n - remain_top )
			{
				memset_normal( src, c, n );
				return src;
			}

			memset_normal( src, c, remain_top );

			unsigned char *src2 = static_cast< unsigned char * >( src ) + remain_top;
			memset_dqword( src2, c, n - remain_top, int2type< ALIGN_DQWORD >() );

			n2 = n - remain_top;
		}

		size_t remain_bottom = n2 % 16;

		if ( remain_bottom )
		{
			unsigned char *src2 = static_cast< unsigned char * >( src ) + ( n - remain_bottom );
			memset_normal( src2, c, remain_bottom );
		}
	}
	else	// 16 > n
	{
		memset_normal( src, c, n );
	}

	return src;
}

/**
 * @brief fast_memcpy
 *
 * @note
 *   dst[n] = src[n]
 */
void *fast_memcpy( void *dst, void *src, size_t n )
{
	if ( 16 <= n )
	{
		const bool dst_aligned = is_dqword_aligned( dst );
		const bool src_aligned = is_dqword_aligned( src );
		size_t n2 = n;

		if ( dst_aligned && src_aligned )
		{
			memcpy_dqword( dst, src, n, int2type< ALIGN_DQWORD >(), int2type< ALIGN_DQWORD >() );
		}
		else
		{
			size_t remain_top = 16 - ( reinterpret_cast<unsigned int>( dst ) % 16 );

			if ( 16 > n - remain_top )
			{
				memcpy_normal( dst, src, n );
				return dst;
			}

			memcpy_normal( dst, src, remain_top );

			unsigned char *dst2 = static_cast< unsigned char * >( dst ) + remain_top;
			unsigned char *src2 = static_cast< unsigned char * >( src ) + remain_top;

			if ( is_dqword_aligned( src2 ) )
				memcpy_dqword( dst2, src2, n - remain_top, int2type< ALIGN_DQWORD >(), int2type< ALIGN_DQWORD >());
			else
				memcpy_dqword( dst2, src2, n - remain_top, int2type< ALIGN_NONE >(), int2type< ALIGN_DQWORD >());

			n2 = n - remain_top;
		}

		size_t remain_bottom = n2 % 16;

		if ( remain_bottom )
		{
			unsigned char *dst2 = static_cast< unsigned char * >( dst ) + ( n - remain_bottom );
			unsigned char *src2 = static_cast< unsigned char * >( src ) + ( n - remain_bottom );
			memcpy_normal( dst2, src2, remain_bottom );
		}
	}
	else	// 16 > n
	{
		memcpy_normal( dst, src, n );
	}

	return dst;
}

/**
 * @brief fast_memccmp
 *
 * @note
 *   return ( src[0] == c && ... src[n] == c ? 0 : 1 );
 */
int fast_memccmp( void *src, int c, size_t n )
{
	int result = 0;

	if ( 16 <= n )
	{
		const bool src_aligned = is_dqword_aligned( src );
		size_t n2 = n;

		if ( src_aligned )
		{
			result = memccmp_dqword( src, c, n, int2type< ALIGN_DQWORD >() );
		}
		else
		{
			size_t remain_top = 16 - ( reinterpret_cast< unsigned int >( src ) % 16 );

			if ( 16 > n - remain_top )
				return memccmp_normal( src, c, n );

			result = memccmp_normal( src, c, remain_top );

			if ( 0 == result )
			{
				unsigned char *src2 = static_cast< unsigned char * >( src ) + remain_top;
				result = memccmp_dqword( src2, c, n - remain_top, int2type< ALIGN_DQWORD >() );

				n2 = n - remain_top;
			}
		}

		if ( 0 == result )
		{
			size_t remain_bottom = n2 % 16;

			if ( remain_bottom )
			{
				unsigned char *src2 = static_cast< unsigned char * >( src ) + ( n - remain_bottom );
				result = memccmp_normal( src2, c, remain_bottom );
			}
		}
	}
	else	// 16 > n
	{
		result = memccmp_normal( src, c, n );
	}

	return result;
}

/**
 * @brief fast_maskccmp
 *
 * @note
 *   return ( src[0] & TRANSPARENT == TRANSPARENT && ... src[n] & TRANSPARENT == TRANSPARENT ? 0 : 1 );
 */
int fast_maskccmp( void *src, size_t n )
{
	int result = 0;

	if ( 16 <= n )
	{
		const bool src_aligned = is_dqword_aligned( src );
		size_t n2 = n;

		if ( src_aligned )
		{
			result = maskccmp_dqword( src, n, int2type< ALIGN_DQWORD >() );
		}
		else
		{
			size_t remain_top = 16 - ( reinterpret_cast< unsigned int >( src ) % 16 );

			if ( 16 > n - remain_top )
				return maskccmp_normal( src, n );

			result = maskccmp_normal( src, remain_top );

			if ( 0 == result )
			{
				unsigned char *src2 = static_cast< unsigned char * >( src ) + remain_top;
				result = maskccmp_dqword( src2, n - remain_top, int2type< ALIGN_DQWORD >() );

				n2 = n - remain_top;
			}
		}

		if ( 0 == result )
		{
			size_t remain_bottom = n2 % 16;

			if ( remain_bottom )
			{
				unsigned char *src2 = static_cast< unsigned char * >( src ) + ( n - remain_bottom );
				result = maskccmp_normal( src2, remain_bottom );
			}
		}
	}
	else	// 16 > n
	{
		result = maskccmp_normal( src, n );
	}

	return result;
}

/**
 * @brief fast_maskcopy
 *
 * @note
 *   ( srcmask[n] & TRANSPARENT ) != TRANSPARENT ---> dst[n] = src[n]
 *   ( srcmask[n] & TRANSPARENT ) == TRANSPARENT ---> no-op
 */
void fast_maskcopy( void *dst, void *src, void *srcmask, size_t n )
{
	if ( 64 <= n )
	{
		const bool tag_aligned = is_dqword_aligned( srcmask );
		const bool src_aligned = is_dqword_aligned( src );
		const bool dst_aligned = is_dqword_aligned( dst );
		size_t n2 = n;

		if ( dst_aligned && src_aligned && tag_aligned )
		{
			maskcopy_64byte( dst, src, srcmask, n, int2type< ALIGN_DQWORD >(), int2type< ALIGN_DQWORD >(), int2type< ALIGN_DQWORD >() );
		}
		else
		{
			size_t remain_top = 16 - ( reinterpret_cast< unsigned int >( srcmask ) % 16 );

			if ( 64 > n - remain_top )
			{
				maskcopy_normal( dst, src, srcmask, n );
				return;
			}

			maskcopy_normal( dst, src, srcmask, remain_top );

			unsigned char *dst2 = static_cast< unsigned char * >( dst ) + remain_top;
			unsigned char *src2 = static_cast< unsigned char * >( src ) + remain_top;
			unsigned char *srcmask2 = static_cast< unsigned char * >( srcmask ) + remain_top;

			if ( is_dqword_aligned( src2 ) && is_dqword_aligned( dst2 ) )
				maskcopy_64byte( dst2, src2, srcmask2, n - remain_top, int2type< ALIGN_DQWORD >(), int2type< ALIGN_DQWORD >(), int2type< ALIGN_DQWORD >() );
			else if ( is_dqword_aligned( src2 ) )
				maskcopy_64byte( dst2, src2, srcmask2, n - remain_top, int2type< ALIGN_DQWORD >(), int2type< ALIGN_DQWORD >(), int2type< ALIGN_NONE >());
			else if ( is_dqword_aligned( dst2 ) )
				maskcopy_64byte( dst2, src2, srcmask2, n - remain_top, int2type< ALIGN_NONE >(), int2type< ALIGN_DQWORD >(), int2type< ALIGN_DQWORD >());
			else
				maskcopy_64byte( dst2, src2, srcmask2, n - remain_top, int2type< ALIGN_NONE >(), int2type< ALIGN_DQWORD >(), int2type< ALIGN_NONE >());

			n2 = n - remain_top;
		}

		size_t remain_bottom = n2 % 64;

		if ( remain_bottom )
		{
			void *dst2 = static_cast< unsigned char * >( dst ) + ( n - remain_bottom );
			void *src2 = static_cast< unsigned char * >( src ) + ( n - remain_bottom );
			void *srcmask2 = static_cast< unsigned char * >( srcmask ) + ( n - remain_bottom );
			maskcopy_normal( dst2, src2, srcmask2, remain_bottom );
		}
	}
	else	// 64 > n
	{
		maskcopy_normal( dst, src, srcmask, n );
	}
}

/**
 * @brief fast_maskset
 *
 * @note
 *   ( srcmask[n] & TRANSPARENT ) != TRANSPARENT ---> dst[n] = c
 *   ( srcmask[n] & TRANSPARENT ) == TRANSPARENT ---> no-op
 */
void fast_maskset( void *dst, int c, void *srcmask, size_t n )
{
	if ( 64 <= n )
	{
		const bool src_aligned = is_dqword_aligned( srcmask );
		const bool dst_aligned = is_dqword_aligned( dst );
		size_t n2 = n;

		if ( dst_aligned && src_aligned )
		{
			maskset_64byte( dst, c, srcmask, n, int2type< ALIGN_DQWORD >(), int2type< ALIGN_DQWORD >() );
		}
		else
		{
			size_t remain_top = 16 - ( reinterpret_cast< unsigned int >( srcmask ) % 16 );

			if ( 64 > n - remain_top )
			{
				maskset_normal( dst, c, srcmask, n );
				return;
			}

			maskset_normal( dst, c, srcmask, remain_top );

			unsigned char *dst2 = static_cast< unsigned char * >( dst ) + remain_top;
			unsigned char *srcmask2 = static_cast< unsigned char * >( srcmask ) + remain_top;

			if ( is_dqword_aligned( dst2 ) )
				maskset_64byte( dst2, c, srcmask2, n - remain_top, int2type< ALIGN_DQWORD >(), int2type< ALIGN_DQWORD >());
			else
				maskset_64byte( dst2, c, srcmask2, n - remain_top, int2type< ALIGN_DQWORD >(), int2type< ALIGN_NONE >());

			n2 = n - remain_top;
		}

		size_t remain_bottom = n2 % 64;

		if ( remain_bottom )
		{
			void *dst2 = static_cast< unsigned char * >( dst ) + ( n - remain_bottom );
			void *srcmask2 = static_cast< unsigned char * >( srcmask ) + ( n - remain_bottom );
			maskset_normal( dst2, c, srcmask2, remain_bottom );
		}
	}
	else	// 64 > n
	{
		maskset_normal( dst, c, srcmask, n );
	}
}
