/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/

#ifndef DEBUG_H
#define DEBUG_H
#if defined(WIN32) || defined(WIN64)
#include <windows.h>
#else
#include <stdio.h>
#endif
#include <string>
#include <exception>
#include <stdarg.h>


namespace AaltoGames{
	namespace Debug
	{
		static void throwError(const char *format,...){
			char c[256];
			va_list params;
			va_start( params, format );     // params to point to the parameter list

			vsprintf_s(c, 256, format, params);
#if defined(WIN32) || defined(WIN64)
			MessageBoxA(0,c,"Exception",MB_OK);
#endif
			throw std::exception(c);
		}
		static void printf(const char *format,...)
		{
#if defined(_DEBUG) || defined(ENABLE_DEBUG_OUTPUT)
			char c[256];
			va_list params;
			va_start( params, format );     // params to point to the parameter list

			vsprintf_s(c, 256, format, params);
#if defined(WIN32) || defined(WIN64)
			OutputDebugStringA(c);
#else
			printf(c);
#endif
#endif
		}
	}
} //AaltoGames

#if defined(_DEBUG) || defined(ENABLE_DEBUG_OUTPUT)
#define AALTO_ASSERT(test, message, ...) if (!(test)) (AaltoGames::Debug::throwError("AALTO_ASSERT ( " #test " ) failed, message: " message, ##__VA_ARGS__),0)
#define AALTO_ASSERT1(test) if (!(test)) (AaltoGames::Debug::throwError("AALTO_ASSERT1 ( " #test " ) failed."),0)
#else
#define AALTO_ASSERT(x, y, ...)
#define AALTO_ASSERT1(x)
#endif

#endif //DEBUG_H
