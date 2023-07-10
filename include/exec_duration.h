#ifndef _EXEC_DURATION_H_
#define _EXEC_DURATION_H_

/*******************************************************************************
 *                                                                              *
 *  Calculate the time elapsed from the start of the first call to the end of   *
 *  the second call.                                                            *
 *  \param verbosity Include detailed information levels in output records.     *
 *  \return Time elapsed between two calls.                                     *
 *                                                                              *
 *******************************************************************************/
extern "C" long exec_duration(int verbosity = 1);

#endif